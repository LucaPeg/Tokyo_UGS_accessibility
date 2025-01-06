import geopandas as gpd
import pandas as pd
import os
import fiona
from collections import defaultdict

# IMPORT LAYERS
data = os.path.join("..\\data\\final\\Tokyo_UGS_accessibility.gpkg")
# print(fiona.listlayers(data))


# Parks catchement areas
accesses = gpd.read_file(data, layer="cleaned_park_accesses")
parks330 = gpd.read_file(data, layer="parks330")
parks660 = gpd.read_file(data, layer="parks660")
parks1000 = gpd.read_file(data, layer="parks1000")  # this boy is like 8 GB

# filter out small parks
parks330 = parks330.query("area >= 30")
parks660 = parks660.query("area >= 30")
parks1000 = parks1000.query("area >= 30")


# Census catchement areas
census = gpd.read_file(data, layer="census_centroids")
# census330 = gpd.read_file(data, layer = 'census330')
# census660 = gpd.read_file(data, layer = 'census660')
# census1000 = gpd.read_file(data, layer = 'census1000')

# fix census datatypes
string_columns = ["KEY_CODE_3", "name_ja", "name_en"]
for col in census.columns:
    if col not in string_columns and col != "geometry":
        census[col] = pd.to_numeric(census[col], errors="coerce", downcast="integer")


########################################################################
# E2SFCA  ##############################################################
########################################################################
# STEP 1: get UGS to population ratios ################################
########################################################################

# Before starting, some checks
# parks330[parks330.geometry.length == 0].count()  # there are 27 parks with no service area
# parks330.geometry.length.describe()

# Associate each park access with the census units it intersects
joined_330 = gpd.sjoin(parks330, census, how="inner", predicate="intersects")
joined_660 = gpd.sjoin(parks660, census, how="inner", predicate="intersects")
joined_1000 = gpd.sjoin(parks1000, census, how="inner", predicate="intersects")

# Initialize the dictionary to store accessible census units for each park
accessibility_dict = {}

# Track which census units have already been assigned to each park
assigned_centroids_per_park = defaultdict(set)
# Step 1: Assign centroids to the smallest travel zone (330m)
for park_id, group in joined_330.groupby("park_id"):
    unique_centroids = group["KEY_CODE_3"].unique()  # Get unique census units
    unassigned_centroids = [
        c for c in unique_centroids if c not in assigned_centroids_per_park[park_id]
    ]  # Exclude already assigned
    if unassigned_centroids:  # Only process parks with unassigned centroids
        park_area = group["area"].iloc[0]
        if pd.isna(park_area):
            park_area = None  # Or handle it as needed, e.g., setting to 0 or skipping the park
        else:
            park_area = int(park_area)  # Convert area to int if valid
        
        accessibility_dict.setdefault(park_id, {}).update(
            {
                "park_area": park_area,  # Add park area
                "330m": unassigned_centroids,
            }
        )
        # Mark these centroids as assigned for this park
        assigned_centroids_per_park[park_id].update(unassigned_centroids)


for park_id, group in joined_330.groupby("park_id"):
    unique_centroids = group["KEY_CODE_3"].unique()  # Get unique census units
    unassigned_centroids = [
        c for c in unique_centroids if c not in assigned_centroids_per_park[park_id]
    ]  # Exclude already assigned
    if unassigned_centroids:  # Only process parks with unassigned centroids
        accessibility_dict.setdefault(park_id, {}).update(
            {
                "park_area": int(group["area"].iloc[0]),  # Add park area
                "330m": unassigned_centroids,
            }
        )
        # Mark these centroids as assigned for this park
        assigned_centroids_per_park[park_id].update(unassigned_centroids)

# Step 2: Assign centroids to the 660m zone (excluding those already assigned in 330m)
for park_id, group in joined_660.groupby("park_id"):
    unique_centroids = group["KEY_CODE_3"].unique()  # Get unique census units
    unassigned_centroids = [
        c for c in unique_centroids if c not in assigned_centroids_per_park[park_id]
    ]  # Exclude already assigned
    if unassigned_centroids:
        accessibility_dict.setdefault(park_id, {}).update(
            {
                "660m": unassigned_centroids,
            }
        )
        # Mark these centroids as assigned for this park
        assigned_centroids_per_park[park_id].update(unassigned_centroids)

# Step 3: Assign centroids to the 1000m zone (excluding those already assigned in 330m and 660m)
for park_id, group in joined_1000.groupby("park_id"):
    unique_centroids = group["KEY_CODE_3"].unique()  # Get unique census units
    unassigned_centroids = [
        c for c in unique_centroids if c not in assigned_centroids_per_park[park_id]
    ]  # Exclude already assigned
    if unassigned_centroids:
        accessibility_dict.setdefault(park_id, {}).update(
            {
                "1000m": unassigned_centroids,
            }
        )
        # Mark these centroids as assigned for this park
        assigned_centroids_per_park[park_id].update(unassigned_centroids)


print(type(list(accessibility_dict.values())[0]))  # Check the type of the first value
print(list(accessibility_dict.values())[0])  # Print the first value
none_count = sum(1 for data in accessibility_dict.values() if data.get('park_area') is None)
parks1000[parks1000['area'].isnull()]["park_id"].nunique()

# GET THE UGS TO POPULATION RATIO
# I will use park area as "supply" of UGS
population_col = "pop_tot"

# Gaussian weights associated with sharp decay (from E2SFCA article, errata corrige)
weights = {
    "330m": 1,
    "660m": 0.44,
    "1000m": 0.03,
}

ugs_population_ratio = {}

# Loop through each park in the accessibility_dict to calculate the UGS-to-population ratio
for park_id, park_data in accessibility_dict.items():
    park_area = park_data.get("park_area")  # Area of the park
    total_weighted_population = 0  # initialize weighted pop

    # Loop through each zone and calculate the weighted population
    for zone, centroids in park_data.items():
        if zone in weights:  # this shouldn't be necessary
            weight = weights[zone]
            # For each census unit in this zone, sum the population and multiply by the zone's weight
            population_sum = 0
            if zone == "330m":
                # For 330m zone, use joined_330
                for centroid in centroids:
                    population_sum += census[census["KEY_CODE_3"] == centroid][
                        population_col
                    ].sum()
            elif zone == "660m":
                # For 660m zone, use joined_660
                for centroid in centroids:
                    population_sum += census[census["KEY_CODE_3"] == centroid][
                        population_col
                    ].sum()
            elif zone == "1000m":
                # For 1000m zone, use joined_1000
                for centroid in centroids:
                    population_sum += census[census["KEY_CODE_3"] == centroid][
                        population_col
                    ].sum()

            # Multiply by the weight for this zone
            total_weighted_population += population_sum * weight

    # Compute the UGS-to-population ratio for this park
    if total_weighted_population > 0:
        ugs_population_ratio[park_id] = park_area / total_weighted_population
    else:
        ugs_population_ratio[park_id] = (
            0  # Handle cases where population might be zero or not assigned
        )

# Now, ugs_population_ratio contains the UGS-to-population ratio for each park
print(ugs_population_ratio)
park_areas = {
    park_id: data["park_area"] for park_id, data in accessibility_dict.items()
}
park_areas

parks330[parks330["area"] == None]

for park_id, data in accessibility_dict.items():
    print(f"Park ID: {park_id}, Park Area Type: {type(data.get('park_area'))}")


none_count = sum(1 for data in accessibility_dict.values() if data["park_area"] is None)
print(f"Number of parks with None for park_area: {none_count}")

print(accessibility_dict.items())
