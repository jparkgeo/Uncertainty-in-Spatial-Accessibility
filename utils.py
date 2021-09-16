import networkx as nx
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point
import numpy as np
import ast
import osmnx as ox
from shapely.ops import cascaded_union
import os
import time


def calculate_min_travel_time(gdf):
    other_road_type = []
    for idx, row in gdf.iterrows():
        # Assign maxspeed for each edge segment
        if row['maxspeed'] is not None:
            if row['maxspeed'].startswith('['):  # if multiple 'maxspeed' (list) is assinged to the edge
                list_speed = ast.literal_eval(row['maxspeed'])
                temp_speed = list_speed[0].split()[0]  # extract only numbers
            else:
                temp_speed = row['maxspeed'].split()[0]  # extract only numbers
        else:
            if row['highway'].startswith('['):
                highway_type = ast.literal_eval(row['highway'])[0]
            else:
                highway_type = row['highway']

            if highway_type == 'motorway':
                temp_speed = 70
            elif highway_type == 'motorway_link':
                temp_speed = 40
            elif highway_type == 'trunk':
                temp_speed = 70
            elif highway_type == 'trunk_link':
                temp_speed = 30
            elif highway_type == 'primary':
                temp_speed = 50
            elif highway_type == 'primary_link':
                temp_speed = 30
            elif highway_type == 'secondary':
                temp_speed = 50
            elif highway_type == 'secondary_link':
                temp_speed = 30
            elif highway_type == 'tertiary':
                temp_speed = 40
            elif highway_type == 'tertiary_link':
                temp_speed = 20
            elif highway_type == 'residential':
                temp_speed = 30
            elif highway_type == 'living_street':
                temp_speed = 20
            elif highway_type == 'unclassified':
                temp_speed = 20
            else:
                temp_speed = 20
                other_road_type.append(row['highway'])

        gdf.at[idx, 'maxspeed'] = int(temp_speed)

    # Calculate time(minute) for passing thorugh the edge segment
    gdf['maxspeed_meters'] = gdf.apply(lambda x: x['maxspeed'] * 26.8223, axis=1)  # convert mile per hour to meter per minute
    gdf['time'] = gdf.apply(lambda x: x['length'] / x['maxspeed_meters'], axis=1)

    if other_road_type:
        print(set(other_road_type))

    gdf = gdf.loc[(gdf['highway'] != 'disused') & (gdf['highway'] != 'razed')]

    return gdf


def populate_uncertainty_with_historical_distribution(gdf, df):
    values = [1, 1.33, 2, 4]
    prob = []

    for val in values:
        val_count = df[df['Class'] == val].shape[0]
        prob.append(round(val_count / df.shape[0], 2))

    if sum(prob) != 1:
        prob[0] = 1 - sum(prob[1:])

    monte = np.random.choice(values, 1, p=prob)
    gdf['traffic_time'] = gdf['time'] * monte
    gdf['traffic_cls'] = monte[0]

    return gdf


def populate_unceertainty_with_random_distribution(gdf):
    values = [1, 1.33]
    prob = [0.75, 0.25]

    monte = np.random.choice(values, gdf.shape[0], p=prob)
    gdf['traffic_time'] = gdf['time'] * monte
    gdf['traffic_cls'] = monte

    return gdf


def road_network_with_uncertainty(file_names, data_path):
    csv_path = data_path.joinpath('traffic_data', 'modified_csv')
    shp_path = data_path.joinpath('traffic_data', 'shp')

    merged_edge = gpd.GeoDataFrame()

    for file_idx, file_name in file_names.iterrows():
        traffic_shp = gpd.read_file(shp_path.joinpath(f'{file_name[0]}.shp'))
        traffic_csv = pd.read_csv(csv_path.joinpath(f'{file_name[0]}.csv'))

        traffic_shp = calculate_min_travel_time(traffic_shp)
        traffic_shp = populate_uncertainty_with_historical_distribution(traffic_shp, traffic_csv)

        merged_edge = merged_edge.append(traffic_shp)

    montorway_no_traffic = gpd.read_file(shp_path.joinpath('motorway_no_traffic.shp'))
    montorway_no_traffic = calculate_min_travel_time(montorway_no_traffic)
    montorway_no_traffic = populate_unceertainty_with_random_distribution(montorway_no_traffic)
    merged_edge = merged_edge.append(montorway_no_traffic)

    non_motorway = gpd.read_file(shp_path.joinpath('non_motorway.shp'))
    non_motorway = calculate_min_travel_time(non_motorway)
    non_motorway = populate_unceertainty_with_random_distribution(non_motorway)
    merged_edge = merged_edge.append(non_motorway)

    return merged_edge


def construct_network(merged_edge, original_nodes):
    # Construct Network
    G = nx.MultiDiGraph()

    # Add Edges
    node_ids = []
    for idx, row in tqdm(merged_edge.iterrows(), total=merged_edge.shape[0]):
        if row['oneway'] == 0:
            G.add_edge(row['to'], row['from'], time=row['time'], traffic_time=row['traffic_time'], length=row['length'])

        G.add_edge(row['from'], row['to'], time=row['time'], traffic_time=row['traffic_time'], length=row['length'])
        node_ids.extend([row['from'], row['to']])

    # Add Nodes
    unique_node_ids = set(node_ids)
    nodes = original_nodes.loc[original_nodes['osmid'].isin(unique_node_ids)]

    for idx, row in tqdm(nodes.iterrows(), total=nodes.shape[0]):
        G.add_node(node_for_adding=row['osmid'], x=row['x'], y=row['y'], highway=row['highway'], ref=row['ref'])

    # Add supplementary attributes
    G.graph['crs'] = merged_edge.crs
    for node, data in G.nodes(data=True):
        data['geometry'] = Point(data['x'], data['y'])

    return G


def remove_uncenessary_nodes(network):
    _nodes_removed = len([n for (n, deg) in network.out_degree() if deg == 0])
    network.remove_nodes_from([n for (n, deg) in network.out_degree() if deg == 0])
    for component in list(nx.strongly_connected_components(network)):
        if len(component) < 10:
            for node in component:
                _nodes_removed += 1
                network.remove_node(node)

    print("Removed {} nodes ({:2.4f}%) from the OSMNX network".format(_nodes_removed, _nodes_removed / float(network.number_of_nodes())))
    print("Number of nodes: {}".format(network.number_of_nodes()))
    print("Number of edges: {}".format(network.number_of_edges()))

    return network


def supply_uncertainty(supply_df, supply_prob):
    values = list(supply_prob.columns)
    probs = list(supply_prob.values[0])
    supply_df['Unc_cls'] = np.random.choice(values, supply_df.shape[0], p=probs)
    supply_df['Unc_ICU'] = supply_df.apply(lambda x: round(x['ADULT_ICU_'] * x['Unc_cls'], 0), axis=1)

    return supply_df


def find_nearest_osm(network, variable):
    for idx, row in tqdm(variable.iterrows(), total=variable.shape[0]):
        if row.geometry.geom_type == 'Point':
            pnt = [row.geometry.y, row.geometry.x]
        elif row.geometry.geom_type == 'Polygon':
            pnt = [row.geometry.centroid.y, row.geometry.centroid.x]
        else:
            print(row.geometry.geom_type)

        nearest_osm = ox.get_nearest_node(network, pnt, method='euclidean')
        variable.at[idx, 'nearest_osm'] = str(nearest_osm)

    return variable


def calculate_catchment_area(network, nearest_osm, minutes, distance_unit='traffic_time'):
    polygons = gpd.GeoDataFrame(crs="EPSG:4326")

    # Create convex hull for each travel time (minutes), respectively.
    for minute in minutes:
        access_nodes = nx.single_source_dijkstra_path_length(network, nearest_osm, minute, weight=distance_unit)
        convex_hull = gpd.GeoSeries(nx.get_node_attributes(network.subgraph(access_nodes), 'geometry')).unary_union.convex_hull
        polygon = gpd.GeoDataFrame({'minutes': [minute], 'geometry': [convex_hull]}, crs="EPSG:4326")
        polygon = polygon.set_index('minutes')
        polygons = polygons.append(polygon)

    # Calculate the differences between convex hulls which created in the previous section.
    for idx, minute in enumerate(minutes):
        if idx != 0:
            current_polygon = polygons.loc[[minute]]
            previous_polygons = cascaded_union(polygons.loc[minutes[:idx], 'geometry'])
            previous_polygons = gpd.GeoDataFrame({'geometry': [previous_polygons]}, crs="EPSG:4326")
            diff_polygon = gpd.overlay(current_polygon, previous_polygons, how="difference")
            if diff_polygon.shape[0] != 0:
                polygons.at[minute, 'geometry'] = diff_polygon['geometry'].values[0]

    return polygons.copy(deep=True)


def E2SFCA_Step1(supply, demand, network, minutes, weights):
    supply['step1'] = np.nan

    for s_idx, s_row in tqdm(supply.iterrows(), total=supply.shape[0]):
        supply_ctmt_area = calculate_catchment_area(network, s_row['nearest_osm'], minutes)

        ctmt_pops = []
        for c_idx, c_row in supply_ctmt_area.iterrows():
            temp_pop = demand.loc[demand.geometry.centroid.within(c_row['geometry'])]['Pop'].sum()
            ctmt_pops.append(temp_pop * weights[c_idx])

        supply.at[s_idx, 'step1'] = 100000 * s_row['Unc_ICU'] / sum(ctmt_pops)

    return supply


def E2SFCA_Step2(supply, demand, network, minutes, weights):
    demand['step2'] = np.nan

    for d_idx, d_row in tqdm(demand.iterrows(), total=demand.shape[0]):
        demand_ctmt_area = calculate_catchment_area(network, d_row['nearest_osm'], minutes)

        ctmt_ratio = []
        for c_idx, c_row in demand_ctmt_area.iterrows():
            temp_ratio = supply.loc[supply.geometry.within(c_row['geometry'])]['step1'].sum()
            ctmt_ratio.append(temp_ratio * weights[c_idx])

        demand.at[d_idx, 'step2'] = sum(ctmt_ratio)

    return demand


def measure_accessibility(_thread_id, supply, demand, supply_prob, file_names, original_nodes, minutes, weights, data_path, result_path):

    np.random.seed(_thread_id * 10000 + int(time.time()))
    
    merged_edge = road_network_with_uncertainty(file_names, data_path)
    G = construct_network(merged_edge, original_nodes)  # Populate uncertainty of supply
    G = remove_uncenessary_nodes(G)
    supply = supply_uncertainty(supply, supply_prob)  # Populate uncertainty of supply

    _supply = E2SFCA_Step1(supply, demand, G, minutes, weights)
    _demand = E2SFCA_Step2(_supply, demand, G, minutes, weights)
    
    temp_result_path = os.path.join(result_path, f'iter_{_thread_id}')
    if not os.path.isdir(temp_result_path):
        os.makedirs(temp_result_path)
    
    _supply.to_file(os.path.join(temp_result_path, f'supply.geojson'), driver='GeoJSON')
    _demand.to_file(os.path.join(temp_result_path, f'demand.geojson'), driver='GeoJSON')

    return result_path


def measure_accessibility_unpacker(args):
    return measure_accessibility(*args)
