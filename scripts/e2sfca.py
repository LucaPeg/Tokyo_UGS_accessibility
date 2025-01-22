import geopandas as gpd
import pandas as pd
import numpy as np
import os
import fiona
from collections import defaultdict
import contextily as ctx
import matplotlib.pyplot as plt

from accessibility_functions import get_accessibility_dict
from accessibility_functions import get_ugs_to_pop_ratios
from accessibility_functions import plot_census_points_with_basemap
from accessibility_functions import get_census_served
from accessibility_functions import get_people_served
from accessibility_functions import get_census_catchment
from accessibility_functions import get_accessibility_index  
from accessibility_functions import plot_parks_with_ratio


# IMPORT LAYERS
data = os.path.join("..\\data\\final\\Tokyo_UGS_accessibility.gpkg")
print(fiona.listlayers(data))

accesses = gpd.read_file(data, layer="park_accesses")
ugs = gpd.read_file(data, layer='ugs')
parks330 = gpd.read_file(data, layer="sa_parks330")
parks660 = gpd.read_file(data, layer="sa_parks660")
parks1000 = gpd.read_file(data, layer="sa_parks1000")

# service areas check
# check length
parks330.geometry.length.describe()
parks330[parks330.geometry.length == 0].count()  # 26 parks with no service area
# technically not an issue, since e2sfca algorithm sorts this out (maybe not optimized)

# check areas
parks330["area"].describe() # these are accesses, so distribution is biased (each access contributes to the quantiles)
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
    #census_list[i] = census_list[i][census_list[i]["pop_tot"] > 15]  # Drop 1st percentile population
    census_list[i] = census_list[i].dropna(subset=["pop_tot"])  # Drop NAs

census, census330, census660, census1000 = census_list

########################################################################
# E2SFCA  ##############################################################
########################################################################
# STEP 1: get UGS to population ratios #################################
# 1.1: all parks #######################################################
########################################################################

# get list of census units for each park
full_accessibility_dict = get_accessibility_dict(  # defined in accessibility_functions.py
    accesses, parks330, parks660, parks1000, census
)
# get for each park its supply over population ratio
ugs_to_pop_ratios = get_ugs_to_pop_ratios(full_accessibility_dict, census)  # same as above

### DIAGNOSTICS #############################################################
# check the distribution
pd.Series(list(ugs_to_pop_ratios.values())).describe()
# distribution is weird, most are around 0 but there are some extremely high values

# I want to extract the parks with highest ugspop ratio and check 
# 1. How many cenusus units they serve
# 2. How many people live in total in those census units
top_parks = sorted(ugs_to_pop_ratios.items(), key=lambda x: x[1], reverse=True)
top_parks = [t[0] for t in top_parks[:20]]
all_parks = list(ugs_to_pop_ratios.keys())
for park in top_parks:
    print(full_accessibility_dict[park])

# does it make sense to remove the census units with few people?
# this increases the UGS to population ratios of some parks (while zeroes some others)
# Another solution would be to eliminate from the accessibility dict the parks that serve 
#   less than a threshold of people (let's say 50) ## actually 1 percentile might make more sense
#   This allows to preserve information about the census units, while tackling the high ugs ratios

census_for_top_parks = get_census_served(top_parks, full_accessibility_dict)
people_for_top_parks = get_people_served(top_parks, full_accessibility_dict, census) # if I run everything this works
people_for_all_parks = get_people_served(all_parks, full_accessibility_dict, census)

pd.Series(list(people_for_all_parks.values())).describe()
pd.Series(list(people_for_all_parks.values())).plot(kind='hist',bins=100)
pd.Series(list(people_for_all_parks.values())).quantile(0.01)
pd.Series(list(people_for_top_parks.values())).describe()

ppl_1percentile = pd.Series(list(people_for_all_parks.values())).quantile(0.01)
# final substep: add UGS_to_pop ratio to "accesses" geodataframe
accesses["ugs_ratio"] = accesses["park_id"].map(ugs_to_pop_ratios)

### FILTERING ###################################################################################
# now I could drop from the accesses datarframe the parks below a certain people-served threshold
# also here I should probably subset the parks to get accessibility for different categories
#################################################################################################

# number of parks that serve less than 713. This is obvious since 713 is the 1st percentile
len({key : value for key, value in people_for_all_parks.items() if value < 713}) # 78 parks

# I filter the 1st percentile of parks by population served
relevant_parks = dict(filter(lambda x: x[1] > ppl_1percentile, people_for_all_parks.items()))
relevant_parks = list(relevant_parks.keys())
accesses = accesses[accesses['park_id'].isin(relevant_parks)] # filter out parks without people | from 17831 to 16000

# PRODUCING DIFFERENT PARK CATEGORIES ###########################################################

# make sure park areas are consistent
area_consistency = accesses.groupby("park_id")["area"].nunique()
inconsistent_parks = area_consistency[area_consistency > 1]
if inconsistent_parks.empty:
    print("The park_area is consistent across all access_ids for each park_id.")
else:
    print("Inconsistent park areas found for the following park_ids:")
    print(inconsistent_parks)
# since area is consistent I can get extract the first area value for each park
parks = ugs[ugs['park_id'].isin(accesses['park_id'])] # exploit the fact that I filtered the accesses already
parks.loc[:,'ugs_ratio'] = parks['park_id'].map(ugs_to_pop_ratios) # map the ratios
parks['area'].plot(kind='hist', bins=100) # there are some issues
parks['area'].describe()
parks.sort_values(by='area', ascending=False)

parks['park_id'].count() # 7679 parks in total
parks[parks['area']>100000]['park_id'].count() # 41 parks above 10^5
parks[parks['area']>10000]['park_id'].count() # 400 parks above 10^4
parks[parks['area']>4000]['park_id'].count() # 843 parks above 4*10^3
parks[parks['area']>1000]['park_id'].count() # 2786 parks above 10^3
parks[parks['area']< 1000]['park_id'].count() # 4893 parks belove 10^3
parks[parks['area']< 750]['park_id'].count() # 4269 parks belove 750 25% reduction drops 12% of observations

parks[parks['area']<100000]['area'].plot(kind='hist',bins=100, title='Histogram of park areas, bins = 100') # distribution is problematic
parks[parks['area']<10000]['area'].plot(kind='hist',bins=100, title='Histogram of park areas, bins = 100')
parks[parks['area']<4000]['area'].plot(kind='hist',bins=100, title='Histogram of park areas, bins = 100')
parks[parks['area']<2000]['area'].plot(kind='hist',bins=100, title='Histogram of park areas, bins = 100')
parks[parks['area']<1000]['area'].plot(kind='hist',bins=100, title='Histogram of park areas, bins = 100')
####################################################################################################
# ARBITRARY DIVISION:  small (0-1k), Medium 1k-4k, Large: 4k-10k, Very large: above 10k 
# OR PERFORM LOGARITHMIC PARTITION (the following)

parks.loc[:,'log_area'] = np.log10(parks['area']) # log transform park areas

# Define boundaries and labels for coloring
labels = ["Very Small", "Small", "Medium", "Large", "Very Large"]
colors = ["#FF9999", "#FFCC99", "#99FF99", "#66CCFF", "#CC99FF"]
size_colors = dict(zip(labels, colors))

# create categorical variable for park size
quantiles_parks = parks['log_area'].quantile([0, 0.25, 0.5, 0.75, 0.95, 1]) # last 5% is to capture the big parks (above 10k)
parks["size_cat"] = pd.cut(parks["log_area"], bins=quantiles_parks, labels=labels, include_lowest=True)
print(parks["size_cat"].value_counts()) # check for safety
parks.groupby("size_cat")['park_id'].count() # first 3 quantiles, then division in very large park

# I add the number of people that can access each park
parks.loc[:,'affluency'] = parks['park_id'].map(people_for_all_parks)

## VISUALIZATION ###########################################################################

## Histogram log(area) by park size category ##
plt.figure(figsize=(8, 6))  # Start a new figure

# loop through each category and plot it on the same axis (otherwise height mismatch)
for category, color in size_colors.items():
    # Filter data for each category
    data = parks[parks['size_cat'] == category]['log_area']
    
    plt.hist( # plot filtered data
        data, 
        bins=100, 
        color=color, 
        alpha=0.7,  # Transparency for better overlap visibility
        label=category,
        range=(parks['log_area'].min(), parks['log_area'].max())  # full data range
    )

plt.xlabel("Log(Park Area)")
plt.ylabel("Frequency")
plt.title("Histogram of Log(Park Area) by Park Size Category")
plt.legend(title="Size Category")
plt.show()  # Display the plot

# Histogram 'area' by park size category #####################################################
plt.figure(figsize=(8, 6))  # Start a new figure

for category, color in size_colors.items():
    data = parks[parks['size_cat'] == category]['area']
    
    plt.hist(
        data, 
        bins=100,  
        color=color, 
        alpha=0.7,  
        label=category,
        range=(parks['area'].min(), 15000)  # full range is problematic
    )

plt.xlabel("Park Area")
plt.ylabel("Frequency")
plt.title("Histogram of Park Area by Park Size Category")
plt.legend(title="Size Category")
plt.show()

# scatterplot of parks (areas x ugs ratio) divided by size_cat
parks[parks['ugs_ratio']>100]['park_id'].nunique()
parks[parks['ugs_ratio']>500]['park_id'].unique()
ratio_limit = 100 # set as you wish for visualization purposes (the following plots)
parks_no_out = parks[parks['ugs_ratio']<ratio_limit]
plt.figure(figsize=(8, 6))
for category, color in size_colors.items():
    subset = parks_no_out[parks_no_out['size_cat'] == category]
    plt.scatter(subset['log_area'], subset['ugs_ratio'], label=category, color=color)

plt.xlabel('log_area')
plt.ylabel('UGS to pop ratio')
plt.title('Scatter Plot by size category')
plt.legend(title='Size category')
plt.show()

# Idea for visualization: add "number of people served" to parks attribute. 
# Then do scatterplot
plt.figure(figsize=(6,8))
for category, color in size_colors.items():
    subset = parks_no_out[parks_no_out['size_cat']==category]
    plt.scatter(subset['affluency'],subset['ugs_ratio'], label=category, color=color, alpha=0.6)

plt.xlabel('Number of people living within 1km of the park accesses')
plt.ylabel("UGS to population ratio")
plt.title("Relation between people living near the park and UGS to population ratio")
plt.legend(title='Size category')
plt.show()

# Most parks have a very low ugs to population ratio.
# See where the parks with the highest ratios are
# Plot over a base map, color by ratio



## DIVISION IN PARK CATEGORIES ###############################################################
# add size category to the accesses gdf

accesses.loc[:, 'size_cat'] = accesses['park_id'].map(parks.set_index('park_id')['size_cat'])

vl_accesses = accesses[accesses['size_cat']=="Very Large"]
lg_accesses = accesses[(accesses['size_cat']== "Very Large") | (accesses['size_cat']=="Large")]
md_accesses = accesses[(accesses['size_cat']== "Very Large") | 
                       (accesses['size_cat']=="Large") |
                       (accesses['size_cat']=="Medium")]
sm_accesses = accesses[(accesses['size_cat']== "Very Large") | 
                       (accesses['size_cat']=="Large") |
                       (accesses['size_cat']=="Medium") |
                       (accesses['size_cat']=="Small")]
all_accesses = accesses.copy()
     
############################################################################################
## STEP 2: for each census unit, sum the ratios of the parks it can access #################
############################################################################################
vl_census_catchements = get_census_catchment(vl_accesses, census330, census660, census1000, census)             
vl_acc_index = get_accessibility_index(vl_census_catchements, census, full_accessibility_dict)

lg_census_catchements = get_census_catchment(lg_accesses, census330, census660, census1000, census)             
lg_acc_index = get_accessibility_index(lg_census_catchements, census, full_accessibility_dict)

md_census_catchements = get_census_catchment(md_accesses, census330, census660, census1000, census)             
md_acc_index = get_accessibility_index(md_census_catchements, census, full_accessibility_dict)

sm_census_catchements = get_census_catchment(sm_accesses, census330, census660, census1000, census)             
sm_acc_index = get_accessibility_index(sm_census_catchements, census, full_accessibility_dict)

all_census_catchements = get_census_catchment(all_accesses, census330, census660, census1000, census)             
all_acc_index = get_accessibility_index(all_census_catchements, census, full_accessibility_dict)

len(vl_census_catchements.keys())

# map the e2sfca index to the census units layer
census["full_ugs_accessibility"] = census["KEY_CODE_3"].map(all_acc_index)
census["vl_ugs_accessibility"] = census["KEY_CODE_3"].map(vl_acc_index)
census["lg_ugs_accessibility"] = census["KEY_CODE_3"].map(lg_acc_index) 
census["md_ugs_accessibility"] = census["KEY_CODE_3"].map(md_acc_index) 
census["sm_ugs_accessibility"] = census["KEY_CODE_3"].map(sm_acc_index)

# evaluate results  
census['vl_ugs_accessibility'].describe() # just 86 observations
census['vl_ugs_accessibility'].plot(kind='hist',bins=100)
# something must be wrong:
list_large_parks = list((vl_accesses['park_id'].unique()))
large_parks_census = get_census_served(list_large_parks, full_accessibility_dict)
unique_units_covered = set()
for census_units in large_parks_census.values():
    unique_units_covered.update(census_units)
len(unique_units_covered)  # more than 4000. What's up with the 86?


census['lg_ugs_accessibility'].describe() # awful distribution
census['lg_ugs_accessibility'].plot(kind='hist',bins=100)

census['md_ugs_accessibility'].describe() # awful distribution
census['md_ugs_accessibility'].plot(kind='hist',bins=100)

census['sm_ugs_accessibility'].describe() # awful distribution
census['sm_ugs_accessibility'].plot(kind='hist',bins=100)

census['full_ugs_accessibility'].describe() # awful distribution
census['full_ugs_accessibility'].plot(kind='hist',bins=100)
census[census['full_ugs_accessibility'] > 2] # most high index have low low population # deal with outliers before normalizing.

# TODO normalize the accessibility values
# TODO fix names issue to have visualizations of amount of greenspace for each ward.
# TODO make sure that the name issue is not a symptom of other attributes issues

# TODO SOLVE ACCESSIBILITY ISSUE: why does vl_ugs_accessibility have only 86 census units?
len(full_accessibility_dict.keys())


accesses.loc[:, 'size_cat'] = accesses['park_id'].map(parks.set_index('park_id')['size_cat'])
