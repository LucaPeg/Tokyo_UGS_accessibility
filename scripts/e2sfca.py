import geopandas as gpd
import pandas as pd
import numpy as np
import os
import fiona
from collections import defaultdict
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point #  to create buffers (check the intersections in get_accessibility_dict)

from accessibility_functions import get_accessibility_dict
from accessibility_functions import get_ugs_to_pop_ratios
from accessibility_functions import plot_census_points_with_basemap
from accessibility_functions import get_census_served
from accessibility_functions import get_people_served
from accessibility_functions import get_census_catchment
from accessibility_functions import get_accessibility_index  
from accessibility_functions import plot_parks_by_ratio
from accessibility_functions import plot_parks_ratio_people 

# IMPORT LAYERS
data = os.path.join("..\\data\\final\\Tokyo_UGS_accessibility.gpkg")
print(fiona.listlayers(data))

study_area_boundary = gpd.read_file(data, layer='tokyo_boundary') # necessary only for plots
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
all_census = gpd.read_file(data, layer='all_census_units') # needed for ugs ratios
census = gpd.read_file(data, layer='internal_census_units') # relevant census units for 2nd step
census330 = gpd.read_file(data, layer="int_census330")
census660 = gpd.read_file(data, layer="int_census660")
census1000 = gpd.read_file(data, layer="int_census1000")

# exploring census dataframe
census["pop_tot"].describe()   # Internal census: 7136 | 1220 after fixing census units
all_census['pop_tot'].describe() # Full census: 8527 | 1155 -> makes sense since I'm adding perifery
census['pop_tot'].quantile(0.01) # 29 internal census
all_census.pop_tot.quantile(0.01)  # 25 full census

# plot difference in the census populaions
plt.hist(all_census["pop_tot"], bins=100, label='All census units')
plt.hist(census["pop_tot"], bins=100, alpha=0.7, label='Internal census units')
plt.xlabel('Population Total')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# where are the 1% census units with the lowest population?
plot_census_points_with_basemap(all_census, "under", all_census.pop_tot.quantile(0.01))
# 
# low population values affect the UGS to population ratios

## I could filter out the census units with low population here
## I decided to trim only the parks with low affluence
## affluence = number of people living within 1km of the park's accesses

########################################################################
# E2SFCA  ##############################################################
########################################################################
# STEP 1: get UGS to population ratios #################################
# 1.1: all parks #######################################################
########################################################################

# get list of census units for each park
full_accessibility_dict = get_accessibility_dict(  # defined in accessibility_functions.py
    accesses, parks330, parks660, parks1000, all_census # all_census to solve issue of boundary parks
)
# get for each park its supply over population ratio (UGS to population ratio)
ugs_to_pop_ratios = get_ugs_to_pop_ratios(full_accessibility_dict, all_census)  # same as above

### DIAGNOSTICS #############################################################
# check the distribution
pd.Series(list(ugs_to_pop_ratios.values())).describe() # median is 0.07, max is 36757 
# distribution is weird, most are around 0 but there are some extremely high values

# I want to extract the parks with highest ugspop ratio and check 
# 1. How many cenusus units they serve
# 2. How many people live in total in those census units
top_parks = sorted(ugs_to_pop_ratios.items(), key=lambda x: x[1], reverse=True)
top_parks = [t[0] for t in top_parks[:20]]
all_parks = list(ugs_to_pop_ratios.keys())

# UGS to population ratios have issues due to low population census units
# I can either trim the census units or the problematic parks
# Eliminating census units does not necessarily fixed the problem
# I decide to eliminate the parks below the 1st percentile of "affluence" (sum of people living within 1km)
census_for_top_parks = get_census_served(top_parks, full_accessibility_dict)
people_for_top_parks = get_people_served(top_parks, full_accessibility_dict, all_census) 
people_for_all_parks = get_people_served(all_parks, full_accessibility_dict, all_census)

pd.Series(list(people_for_all_parks.values())).describe()
pd.Series(list(people_for_all_parks.values())).plot(kind='hist',bins=100)
pd.Series(list(people_for_all_parks.values())).quantile(0.01) #2866
pd.Series(list(people_for_top_parks.values())).describe() # all values are now below the 1st percentile value population wise

ppl_1percentile = pd.Series(list(people_for_all_parks.values())).quantile(0.01) # 2866
# final substep: add UGS_to_pop ratio to "accesses" geodataframe
accesses["ugs_ratio"] = accesses["park_id"].map(ugs_to_pop_ratios)

# check where the parks with highest ratio are
parks = ugs[ugs['park_id'].isin(accesses['park_id'])] 
parks = parks.copy()
parks.loc[:,'ugs_ratio'] = parks['park_id'].map(ugs_to_pop_ratios) # map the ratios
# I add the number of people that can access each park
parks.loc[:, 'affluency'] = parks['park_id'].map(people_for_all_parks)

### FILTERING ###################################################################################
# now I could drop from the accesses datarframe the parks below a certain people-served threshold
# also here I should probably subset the parks to get accessibility for different categories
#################################################################################################

# I filter the 1st percentile of parks by population served
relevant_parks = dict(filter(lambda x: x[1] > ppl_1percentile, people_for_all_parks.items())) # switch 713 with ppl_1percentile
relevant_parks = list(relevant_parks.keys())
accesses = accesses[accesses['park_id'].isin(relevant_parks)] # filter out parks without people | from 17831 to 16000
parks = parks[parks['park_id'].isin(relevant_parks)] # filter out the parks with low affluence 

## UGS to population ratio DIAGNOSTICS ########################################################
# this is after removing the 1% of parks with the lowest "affluence" (i.e., number of people living in their surroundings)
parks['ugs_ratio'].describe() # from 8241 parks to 8158 parks. median still 0.07, max 457 (it was 36757)
parks.affluency.describe()
parks['ugs_ratio'].plot(kind='hist', bins=100)
parks['ugs_ratio'].plot(kind='hist', bins=100, range=(0,1.5))
parks[parks['ugs_ratio']<1.5]['park_id'].nunique() # 7752 parks out of 8158 have ratio below 1.5
parks_high_ratio = parks[parks['ugs_ratio']>1.5]['park_id'].unique() # 406

#################################################################################################
# PRODUCING DIFFERENT PARK CATEGORIES ###########################################################
#################################################################################################
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
parks.loc[:, 'log_area'] = np.log10(parks['area']) # log transform park areas

# Define boundaries and labels for coloring
labels = ["Very Small", "Small", "Medium", "Large", "Very Large"]
colors = ["#FF9999", "#FFCC99", "#99FF99", "#66CCFF", "#CC99FF"]
size_colors = dict(zip(labels, colors))

# create categorical variable for park size
quantiles_parks = parks['log_area'].quantile([0, 0.25, 0.5, 0.75, 0.95, 1]) # last 5% is to capture the big parks (above 10k)
parks = parks.copy()
parks.loc[:, "size_cat"] = pd.cut(parks["log_area"], bins=quantiles_parks, labels=labels, include_lowest=True)
print(parks["size_cat"].value_counts()) # check for safety
parks.groupby("size_cat")['park_id'].count() # first 3 quantiles, then division in very large park

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
## VISUALIZATION ###########################################################################
############################################################################################
# TODO fix names issue to have visualizations of amount of greenspace for each ward.

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
parks[parks['ugs_ratio']>100]['park_id'].unique()
ratio_limit = 2 # set as you wish for visualization purposes (the following plots)
parks_no_out = parks[parks['ugs_ratio']<ratio_limit] # remove outliers

plt.figure(figsize=(8, 6))
for category, color in size_colors.items():
    subset = parks_no_out[parks_no_out['size_cat'] == category]
    plt.scatter(subset['log_area'], subset['ugs_ratio'], label=category, color=color)

plt.xlabel('log_area')
plt.ylabel('UGS to pop ratio')
plt.title('Scatter Plot by size category')
plt.legend(title='Size category')
plt.show()

# plot relationship between 'affluence' and ugs to population ratio 
plt.figure(figsize=(6,8))
for category, color in size_colors.items():
    subset = parks_no_out[parks_no_out['size_cat']==category]
    plt.scatter(subset['affluency'],subset['ugs_ratio'], label=category, color=color, alpha=0.6)

plt.xlabel('Number of people living within 1km of the park accesses')
plt.ylabel("UGS to population ratio")
plt.title("Relation between people living near the park and UGS to population ratio")
plt.legend(title='Size category')
plt.show()

# VISUALIZE TROUBLESOME PARKS #TODO solve the high ugs to pop ratio issue
plot_parks_by_ratio(parks, 1.5, study_area_boundary) 
plot_parks_by_ratio(parks, 10, study_area_boundary)
plot_parks_by_ratio(parks, 100, study_area_boundary) 

plot_parks_ratio_people(parks, 1.5, study_area_boundary)
plot_parks_ratio_people(parks, 10, study_area_boundary)
plot_parks_ratio_people(parks, 100, study_area_boundary)

# Two categories of "problematic" parks: 
# 1. large parks in non residential areas 
# 2. Riversides -> large and few people in the surrounding. Adding other variables to park supply would lower their ratios (facilities, length of trails inside parks)
# 3. Parks at the edges of Tokyo (because I do not consider the census unit outside the study area in the UGS_ratio computation)
     
# TODO Deal with outliers before moving to the second step
# outliers are mainly Koto area and riversides
     
############################################################################################
## STEP 2: for each census unit, sum the ratios of the parks it can access #################
############################################################################################

vl_census_catchements = get_census_catchment(vl_accesses, census330, census660, census1000, census)  # census are the internal units           
vl_acc_index = get_accessibility_index(vl_census_catchements, census, full_accessibility_dict)

lg_census_catchements = get_census_catchment(lg_accesses, census330, census660, census1000, census)             
lg_acc_index = get_accessibility_index(lg_census_catchements, census, full_accessibility_dict)

md_census_catchements = get_census_catchment(md_accesses, census330, census660, census1000, census)             
md_acc_index = get_accessibility_index(md_census_catchements, census, full_accessibility_dict)

sm_census_catchements = get_census_catchment(sm_accesses, census330, census660, census1000, census)             
sm_acc_index = get_accessibility_index(sm_census_catchements, census, full_accessibility_dict)

all_census_catchements = get_census_catchment(all_accesses, census330, census660, census1000, census)             
all_acc_index = get_accessibility_index(all_census_catchements, census, full_accessibility_dict)


# map the e2sfca index to the census units layer
census["full_ugs_accessibility"] = census["KEY_CODE_3"].map(all_acc_index)
census["vl_ugs_accessibility"] = census["KEY_CODE_3"].map(vl_acc_index)
census["lg_ugs_accessibility"] = census["KEY_CODE_3"].map(lg_acc_index) 
census["md_ugs_accessibility"] = census["KEY_CODE_3"].map(md_acc_index) 
census["sm_ugs_accessibility"] = census["KEY_CODE_3"].map(sm_acc_index)

# evaluate results  
census['vl_ugs_accessibility'].describe()
census['vl_ugs_accessibility'].plot(kind='hist',bins=100)
census[census['vl_ugs_accessibility']>40]['KEY_CODE_3'].nunique() # 90 units with unusually high values
census['vl_ugs_accessibility'].plot(kind='hist',bins=100, range=(0,40))

census['lg_ugs_accessibility'].describe() 
census['lg_ugs_accessibility'].plot(kind='hist',bins=100)

census['md_ugs_accessibility'].describe() 
census['md_ugs_accessibility'].plot(kind='hist',bins=100)

census['sm_ugs_accessibility'].describe() 
census['sm_ugs_accessibility'].plot(kind='hist',bins=100)

census['full_ugs_accessibility'].describe()
census['full_ugs_accessibility'].plot(kind='hist',bins=100)


# normalize the values: min max 
acc_score_list = ['full_ugs_accessibility',
                  'vl_ugs_accessibility',
                  'lg_ugs_accessibility',
                  'md_ugs_accessibility',
                  'sm_ugs_accessibility'
                  ]
for acc_score in acc_score_list:
    min_val = census[acc_score].min()
    max_val = census[acc_score].max()
    census[acc_score] = (census[acc_score] - min_val) / max_val


# TODO Merge the actual polygonal geometry for visualization purposes
# TODO Perform GWR