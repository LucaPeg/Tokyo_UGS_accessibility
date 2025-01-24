# THIS SCRIPT IS OBSOLETE

# I used this script to check whether park accesses and census centroids were correctly snapped onto the road network.
# However, after cleaning them I delete the original file from which I started the cleaning process
# The accesses find in the final geopackage are clean (it can be checked)
# The accesses employed in previous iterations of the research can be found in the "provisional" folder

# THIS SCRIPT IS TO BE INTENDED AS PERSONAL REFERENCE
# TODO rewrite the script starting from the new 'raw' accesses (which can be obtained through QGIS)
# TODO obtain the accesses directly with Python (I tried but it was inefficient)

import geopandas as gpd
import os
from shapely.strtree import STRtree
import fiona
import gdown
import pandas as pd

# Define a function to get the distance between an access point and its closest road
def distance_to_nearest_road(point, spatial_index):
    nearby_indices = spatial_index.query(point.buffer(50)) 
    nearby_roads = roads["geometry"].iloc[nearby_indices]
    if len(nearby_roads) > 0:
        nearest_geom = min(nearby_roads, key=lambda road: point.distance(road))
        return point.distance(nearest_geom)
    else:
        return float('inf')  # Infinite distance if no nearby roads found
    
# data import
data = os.path.join("..\\data\\final\\Tokyo_UGS_accessibility.gpkg")
ugs = gpd.read_file(data, layer = 'ugs')
roads = gpd.read_file(data, layer = 'road_network')
roads_tree = STRtree(roads["geometry"].values) # creates a spatial index
road_sindex = roads.sindex
census_centroids = gpd.read_file(data, layer ='census_centroids') # these are only internal centroids
accesses = gpd.read_file(data, layer = 'access_points_merged_parks')
all_census_centroids = gpd.read_file(data, layer='all_census_units')
all_census_centroids["distance_to_road"] = all_census_centroids["geometry"].apply(lambda point: distance_to_nearest_road(point, roads_tree))

# If I consider all census centroids, how many are not correctly snapped onto roads?
far_all_census = all_census_centroids[all_census_centroids['distance_to_road'] > 1]
print(f"There are {len(far_all_census)} census centroids that are not on roads")

old_accesses = gpd.read_file(data, layer = 'cleaned_park_accesses')
old_accesses.columns

accesses["id_access"] = range(1,len(accesses)+1)
accesses = accesses[['area', 'CODE5', 'NAME1', 'NAME2', 'park_id', 'id_access', 'geometry']].copy()

# Apply to the accesses GeoDataFrame
accesses["distance_to_road"] = accesses["geometry"].apply(lambda point: distance_to_nearest_road(point, roads_tree))
census_centroids["distance_to_road"] = census_centroids["geometry"].apply(lambda point: distance_to_nearest_road(point, roads_tree))

far_accesses = accesses[accesses['distance_to_road'] > 1]
far_census = census_centroids[census_centroids['distance_to_road'] > 1]

print(f"There are {len(far_accesses)} access points that are not on roads")
print(f"There are {len(far_census)} census centroids that are not on roads")

far_accesses['park_id'].unique()

# I want to find what parks have all their accesses in "far" accesses 
# each access has a park_id and an id_access
parks_remote_access = far_accesses['park_id'].unique().tolist()

# for parks with remote accessess, find how many
far_accesses.groupby("park_id")["id_access"].count()

# create dictionary with park id and number of remote accesses
park_remote_accesses_counts = far_accesses.groupby("park_id")["id_access"].count().to_dict()
park_accesses_counts = accesses.groupby("park_id")['id_access'].count().to_dict()    

# find parks that have solely remote accesses (and thus are remote parks)
remote_parks = []
for park_id, count in park_remote_accesses_counts.items():
    total_accesses = park_accesses_counts[park_id]
    non_remote_accesses = total_accesses - count
    if non_remote_accesses == 0:
        remote_parks.append(park_id)
        

parks_no_access = ugs[ugs["park_id"].isin(remote_parks)] 
# parks_no_access.to_file('remote_parks.geojson') ## visualized in QGIS: I decided to remove them

final_accesses = accesses[accesses['distance_to_road'] < 1]
print(f"Total number of accesses: {accesses.shape[0]}")
print(f"Final number of accesses: {final_accesses.shape[0]}")
print(f"Remote accesses: {far_accesses.shape[0]}")

# make sure I actually removed the correct thing
assert accesses.shape[0]- final_accesses.shape[0] == far_accesses.shape[0]

# export the final accesses:
#final_accesses.to_file(data, layer="cleaned_merged_park_accesses", driver="GPKG", index=False)
# I actually changed the name of the accesses

assert final_accesses['id_access'].is_unique, "Duplicate id_access values found!"
assert final_accesses.columns.is_unique, "Column names are not unique!"

#final_accesses = final_accesses.rename(columns=lambda x: x.replace(" ", "_").replace("-", "_"))