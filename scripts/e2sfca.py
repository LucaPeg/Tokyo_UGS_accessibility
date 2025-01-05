import geopandas as gpd
import pandas as pd
import os
import fiona

# IMPORT LAYERS
data = os.path.join("..\\data\\final\\Tokyo_UGS_accessibility.gpkg")
print(fiona.listlayers(data))


# Parks catchement areas
accesses = gpd.read_file(data,layer = 'cleaned_park_accesses')
parks330 = gpd.read_file(data, layer = 'parks330')
parks660 = gpd.read_file(data, layer = 'parks660') 
parks1000 = gpd.read_file(data, layer = 'parks1000') # this boy is like 8 GB

# Census catchement areas
census = gpd.read_file(data, layer = 'census_centroids')
#census330 = gpd.read_file(data, layer = 'census330')
#census660 = gpd.read_file(data, layer = 'census660')
#census1000 = gpd.read_file(data, layer = 'census1000')

########################################################################
# E2SFCA  ##############################################################
########################################################################
# STEP 1: get UGS to population ratios ##############################
########################################################################

# Before starting, some checks
parks330[parks330.geometry.length == 0].count() # there are 27 parks with no service area
parks330.geometry.length.describe()

# Associate each park access with the census units it intersects
joined_330 = gpd.sjoin(parks330, census, how="inner", predicate="intersects")
joined_660 = gpd.sjoin(parks660, census, how="inner", predicate="intersects")
joined_1000 = gpd.sjoin(parks1000, census, how="inner", predicate="intersects")

results = {"330m":{}, "660m":{}, "1000m":{}}

## I FEEL LIKE THE FOLLOWING IS WRONG BECAUSE CENTROIDS GET ASSIGNED ONLY ONCE. 
## HOWEVER I WANT CENTROIDS TO BE ASSIGNED ONLY ONCE TO A SINGLE PARK, BUT TO MULTIPLE PARKS

# Step 1: Assign centroids to the 330m zone
assigned_centroids = set() # track which centroids I assigned already
for park_id, group in joined_330.groupby("park_id"):
    # Avoid duplicate centroids reached by multiple access points
    unique_centroids = group["KEY_CODE_3"].unique()
    results["330m"][park_id] = {
        "park_area": float(group["area"].iloc[0]),
        "census_units": list(unique_centroids),
    }
    assigned_centroids.update(unique_centroids)


# Step 2: Assign centroids to the 660m zone (excluding already assigned centroids)
for park_id, group in joined_660.groupby("park_id"):
    unique_centroids = group["KEY_CODE_3"].unique()
    remaining_centroids = [c for c in unique_centroids if c not in assigned_centroids]
    if remaining_centroids:  # Only add parks with remaining centroids
        results["660m"][park_id] = {
            "park_area": float(group["area"].iloc[0]),
            "census_units": remaining_centroids,
        }
        assigned_centroids.update(remaining_centroids)

# Step 3: Assign centroids to the 1000m zone (excluding already assigned centroids)
for park_id, group in joined_1000.groupby("park_id"):
    unique_centroids = group["KEY_CODE_3"].unique()
    remaining_centroids = [c for c in unique_centroids if c not in assigned_centroids]
    if remaining_centroids:  # Only add parks with remaining centroids
        results["1000m"][park_id] = {
            "park_area": float(group["area"].iloc[0]),
            "census_units": remaining_centroids,
        }

    
for park_id, group in joined_660.groupby("park_id"):
    if park_id == 1504:
        print(f"Park ID: {park_id}")
        print(group[['KEY_CODE_3','area']])
        print("\n" + "-"*50 + "\n")
        
        