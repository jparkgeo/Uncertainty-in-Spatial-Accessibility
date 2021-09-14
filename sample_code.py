import geopandas as gpd
import pandas as pd
import pathlib
import multiprocessing as mp
import itertools
import utils
import time
import os


data_path = pathlib.Path(r'./sample_data')
csv_path = data_path.joinpath('traffic_data', 'modified_csv')
shp_path = data_path.joinpath('traffic_data', 'shp')
result_path = os.getenv('result_folder')
processor_num = 4

assert data_path
assert result_path


# Import mobility-related files
file_names = pd.read_csv(data_path.joinpath('traffic_data', 'file_names.txt'), header=None)
original_nodes = gpd.read_file(shp_path.joinpath('nodes.shp'))
merged_edge = utils.road_network_with_uncertainty(file_names, data_path)
G = utils.construct_network(merged_edge, original_nodes)
G = utils.remove_uncenessary_nodes(G)

# Load ICU bed availability data
xls_file = pd.read_csv(data_path.joinpath('hospital_availability.csv'))
xls_file = xls_file.loc[xls_file['County'] == f'Harris']
xls_file = xls_file.set_index('Date')

# Percentage of available ICU beds per day
avail_ICU = pd.DataFrame(index=xls_file.index, columns=['av_ICU'])
avail_ICU['av_ICU'] = xls_file.apply(lambda x: (x['Avail_ICU'] + x['COV_S_ICU'] + x['COV_C_ICU']) / 1614, axis=1)

# Calculate the percentage of how many percentage of ICU beds are available
probs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
supply_prob = pd.DataFrame(index=['av_ratio'], columns=probs)
for idx, val in enumerate(probs):
    if idx == len(probs) - 1:
        break

    supply_prob.loc['av_ratio', probs[idx+1]] = avail_ICU.loc[(avail_ICU['av_ICU'] >= val) & (avail_ICU['av_ICU'] < probs[idx +1])].shape[0]
    supply_prob[val] = supply_prob[val].astype(float)

supply_prob = supply_prob / 153  # 153 days
supply_prob = supply_prob.drop(columns=[0.0])
supply_prob = supply_prob.round(decimals=2)

# Import supply (ICU beds) and demand (residential population)
supply = gpd.read_file(data_path.joinpath('supply_sample.shp'))
supply = supply.set_index('SupplyID')
supply = supply.loc[supply['ADULT_ICU_'] != 0]

demand = gpd.read_file(data_path.joinpath('demand_sample.shp'))
demand = demand.set_index('GRID_ID')
demand = demand.loc[demand['Note'] != 'water']

# Find nearest node of OSM from supply and demand locations
supply = utils.find_nearest_osm(G, supply)
demand = utils.find_nearest_osm(G, demand)

# Set threshold travel time and corresponding spatial impedance
# minutes = [10, 20, 30]
minutes = [10]
weights = {10: 1, 20: 0.68, 30: 0.22}

# Measure accessibility with multiprocessing package
start = int(time.time())
pool = mp.Pool(processes = processor_num)
access_result = pool.map(utils.measure_accessibility_unpacker,
                         zip(range(processor_num),
                             itertools.repeat(supply),
                             itertools.repeat(demand),
                             itertools.repeat(supply_prob),
                             itertools.repeat(file_names),
                             itertools.repeat(original_nodes),
                             itertools.repeat(minutes),
                             itertools.repeat(weights),
                             itertools.repeat(data_path),
                             itertools.repeat(result_path)
                            )
                        )
end = int(time.time())
pool.close()
print("***run time(min) : ", (end-start)/60)

# Save results
for i in range(processor_num):
    access_result[i][0].to_file(os.path.join(result_path, f'supply_{i}.geojson'), driver='GeoJSON')
#     access_result[i][0].to_file(result_path.joinpath(f'supply_{i}.shp'))
    access_result[i][1].to_file(os.path.join(result_path, f'demand_{i}.geojson'), driver='GeoJSON')
#     access_result[i][1].to_file(result_path.joinpath(f'demand_{i}.shp'))
    
   
