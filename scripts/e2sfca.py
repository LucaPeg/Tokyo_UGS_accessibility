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

# filter out small parks (the merging process created some zero area entries)
parks330 = parks330.query("area >= 30")
parks660 = parks660.query("area >= 30")
parks1000 = parks1000.query("area >= 30")

# Census catchement areas
census = gpd.read_file(data, layer="census_centroids")
# census330 = gpd.read_file(data, layer = 'census330')
# census660 = gpd.read_file(data, layer = 'census660')
# census1000 = gpd.read_file(data, layer = 'census1000')

# fix census datatypes (they are almost all 'objects' otherwise)
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

# Group the accesses layer by park_id and fetch the area for each park
# Otherwise I get "None" as area value for 1200 parks. Even after filtering the ones without intersections.

park_areas_from_access = accesses.groupby("park_id")["area"].first().to_dict()
# Track which census units have already been assigned to each park
assigned_centroids_per_park = defaultdict(set)
# initialize the dictionary that will contain the UGS-census unit relation
accessibility_dict = {}
# I add all parks. Previously I loaded the park areas in the 330m loop but it was incorrect
# I got area = None for parks that had no intersections in the 330m area but had some in 660m or 1000m
for park_id in park_areas_from_access:
    accessibility_dict[park_id] = {
        "park_area": int(park_areas_from_access[park_id]),
        "330m": [],
        "660m": [],
        "1000m": []
    }

# Now process the 330m service area and update the parks with intersections
for park_id, group in joined_330.groupby("park_id"):
    unique_centroids = group["KEY_CODE_3"].unique()
    unassigned_centroids = [
        c for c in unique_centroids if c not in assigned_centroids_per_park[park_id]
    ]
    if unassigned_centroids:
        park_area = group["area"].iloc[0]
        if pd.isna(park_area):
            park_area = None
        else:
            park_area = int(park_area)
        
        accessibility_dict[park_id]["park_area"] = park_area  # Update park area
        accessibility_dict[park_id]["330m"].extend(unassigned_centroids)
        assigned_centroids_per_park[park_id].update(unassigned_centroids)

# Now process the 660m service area (excluding those already assigned in 330m)
for park_id, group in joined_660.groupby("park_id"):
    unique_centroids = group["KEY_CODE_3"].unique()
    unassigned_centroids = [
        c for c in unique_centroids if c not in assigned_centroids_per_park[park_id]
    ]
    if unassigned_centroids:
        accessibility_dict[park_id]["660m"].extend(unassigned_centroids)
        assigned_centroids_per_park[park_id].update(unassigned_centroids)

# Finally, process the 1000m service area (excluding those already assigned in 330m and 660m)
for park_id, group in joined_1000.groupby("park_id"):
    unique_centroids = group["KEY_CODE_3"].unique()
    unassigned_centroids = [
        c for c in unique_centroids if c not in assigned_centroids_per_park[park_id]
    ]
    if unassigned_centroids:
        accessibility_dict[park_id]["1000m"].extend(unassigned_centroids)
        assigned_centroids_per_park[park_id].update(unassigned_centroids)

none_count = sum(1 for data in accessibility_dict.values() if data.get('park_area') is None)
assert none_count == 0, "There are null values as park area value"

#However a lot of parks have zero intersections in all the travel time zones.
print(f"There are in {len(accessibility_dict.keys())} parks in the accessibility dictonary")

# I filter out all the parks that have zero intersections
filtered_accessibility_dict = {
    park_id: data for park_id, data in accessibility_dict.items()
    if data["330m"] or data["660m"] or data["1000m"]
}

print(f"{len(accessibility_dict.keys())- len(filtered_accessibility_dict.keys())} parks without interesctions removed")


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
for park_id, park_data in filtered_accessibility_dict.items():
    park_area = park_data.get("park_area")  # Area of the park
    total_weighted_population = 0  # initialize weighted pop

    # Loop through each zone and calculate the weighted population
    for zone, centroids in park_data.items():
        if zone in weights:  # this shouldn't be necessary
            weight = weights[zone]
            # For each census unit in this zone, sum the population and multiply by the zone's weight
            population_sum = 0
            if zone == "330m":
                for centroid in centroids:
                    population_sum += census[census["KEY_CODE_3"] == centroid][
                        population_col
                    ].sum()
            elif zone == "660m":
                for centroid in centroids:
                    population_sum += census[census["KEY_CODE_3"] == centroid][
                        population_col
                    ].sum()
            elif zone == "1000m":
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

print(ugs_population_ratio)
# check the distribution
pd.Series(list(ugs_population_ratio.values())).describe()
pd.Series(list(ugs_population_ratio.values())).plot()
#distribution is weird, most are around 0 but there are some extremely high values

# get the largest ugs_population_ratio
sorted(ugs_population_ratio.items(), key=lambda x: x[1], reverse=True)

# get the set of the census units intersected by the parks with the highest ugs to pop ratio
int_census_units = set()
for park_id, data in filtered_accessibility_dict.items():
    if park_id == 8200: #highest ugs_to_pop ratio
        for zone in ['330m','660m','1000m']:
            int_census_units.update(data[zone])

# check the population -> I think here is where the problem lies
for code in int_census_units:
    print(census[census['KEY_CODE_3'] == code]['pop_tot']) # just one person lives there
# if 1 person lives in the 1k catchmen, the UGS to pop ratio is: Area/0.03 -> skyrockets

# a solution would be to drop the census units below a population threshold.
# explore the census data
census['pop_tot'].describe() # 5818 census units, mean 1252 people
census['pop_tot'].plot(kind='hist', bins=100)
census[census['pop_tot']>3000]["KEY_CODE_3"].nunique() # 41 census units above 4000
census[census['pop_tot']>4000]["KEY_CODE_3"].nunique() # 3 census units above 4000
census[census['pop_tot']<10]["KEY_CODE_3"].nunique() # 43 census units below 100
census[census['pop_tot']<50]["KEY_CODE_3"].nunique() # 109 census units below 100
census[census['pop_tot']<100]["KEY_CODE_3"].nunique() # 155 census units below 100

# filter the