import geopandas as gpd
import pandas as pd
import os
import fiona
from collections import defaultdict
import contextily as ctx
import matplotlib.pyplot as plt

from accessibility_functions import get_accessibility_dict
from accessibility_functions import get_ugs_to_pop_ratios
from accessibility_functions import plot_census_points_with_basemap


# IMPORT LAYERS
data = os.path.join("..\\data\\final\\Tokyo_UGS_accessibility.gpkg")
print(fiona.listlayers(data))

accesses = gpd.read_file(data, layer="park_accesses")
parks330 = gpd.read_file(data, layer="sa_parks330")
parks660 = gpd.read_file(data, layer="sa_parks660")
parks1000 = gpd.read_file(data, layer="sa_parks1000")

# service areas check
# check length
parks330.geometry.length.describe()
parks330[parks330.geometry.length == 0].count()  # 26 parks with no service area
# technically not an issue, since e2sfca algorithm sorts this out (maybe not optimized)

# check areas
parks330["area"].describe()
parks330[parks330["area"] <= 30].count()  # 87 parks below 30sqm
# very small parks (assume below 30sqm) are due to merging process errors

# filter out small parks (the merging process created some zero area entries)
parks330 = parks330.query("area >= 30")
parks660 = parks660.query("area >= 30")
parks1000 = parks1000.query("area >= 30")

# Census catchement areas
census = gpd.read_file(data, layer="census_centroids")
census330 = gpd.read_file(data, layer="census330")
census660 = gpd.read_file(data, layer="census660")
census1000 = gpd.read_file(data, layer="census1000")

# fix census datatypes (they are almost all 'objects')
census_list = [census, census330, census660, census1000]
string_columns = ["KEY_CODE_3", "name_ja", "name_en"]
for col in census.columns:
    if col not in string_columns and col != "geometry":
        for i in range(len(census_list)):
            census_list[i][col] = pd.to_numeric(
                census_list[i][col], errors="coerce", downcast="integer"
            )

# exploring census dataframe
census["pop_tot"].describe()  # 5818 census units, mean 1252 people
census["pop_tot"].plot(kind="hist", bins=100)
census.pop_tot.quantile(0.01)  # 1 percentile is 14.17 people per census unit
census.pop_tot.quantile(0.99)  # 99 percentile is 2830
census.pop_tot.isna().sum()  # 269 NAs
plot_census_points_with_basemap(census, "under", threshold=15)
plot_census_points_with_basemap(census, "over", 3500)
# low population values affect the UGS to population ratios

## MAYBE I SHOULD SKIP THIS AND JUST DROP THE PARKS WITH FEW PEOPLE SERVED
# I trim out the census points below the 1st percentile
for i in range(len(census_list)):
    census_list[i] = census_list[i][census_list[i]["pop_tot"] > 15]  # Drop 1st percentile population
    census_list[i] = census_list[i].dropna(subset=["pop_tot"])  # Drop NAs

census, census330, census660, census1000 = census_list

# TODO apply accessibility function to different subset of parks (by size)
# TODO add the UGS to population ratio to each park access
# there are census units with very low population count
# low values in pop affect the ugs to population ratio


########################################################################
# E2SFCA  ##############################################################
########################################################################
# STEP 1: get UGS to population ratios ################################
########################################################################
accessibility_dict = get_accessibility_dict(  # defined in accessibility_functions.py
    accesses, parks330, parks660, parks1000, census
)

ugs_to_pop_ratios = get_ugs_to_pop_ratios(accessibility_dict, census)  # same as above

# check the distribution
pd.Series(list(ugs_to_pop_ratios.values())).describe()
pd.Series(list(ugs_to_pop_ratios.values())).plot()
# distribution is weird, most are around 0 but there are some extremely high values

# I want to extract the parks with highest ugspop ratio and check 
# 1. How many cenusus units they serve
# 2. How many people live in total in those census units
top_parks = sorted(ugs_to_pop_ratios.items(), key=lambda x: x[1], reverse=True)
top_parks = [t[0] for t in top_parks]

for park in top_parks:
    print(accessibility_dict[park])
    
# does it make sense to remove the census units with few people?
# This increases the UGS to population ratios of some parks (while zeroes some others)
# Another solution would be to eliminate from the accessibility dict the parks that serve 
#   less than a threshold of people (let's say 50)
#   This allows to preserve information about the census units, while tackling the high ugs ratios

from accessibility_functions import get_census_served
from accessibility_functions import get_people_served 

census_for_each_park = get_census_served(top_parks, accessibility_dict)
people_for_each_park = get_people_served(top_parks, accessibility_dict, census) # if I run everything this works
# why was it giving problems before, then?
pd.Series(list(people_for_each_park.values())).describe()
pd.Series(list(people_for_each_park.values())).quantile(0.01)


############################################################################################
## STEP 2: for each census unit, sum the ratios of the parks it can access #################
############################################################################################

