import geopandas as gpd
import pandas as pd
import os
import fiona
from collections import defaultdict
import contextily as ctx
import matplotlib.pyplot as plt

from accessibility_functions import get_accessibility_dict
from accessibility_functions import get_ugs_to_pop_ratios
# IMPORT LAYERS
data = os.path.join("..\\data\\final\\Tokyo_UGS_accessibility.gpkg")
print(fiona.listlayers(data))

accesses = gpd.read_file(data, layer='park_accesses')
parks330 = gpd.read_file(data, layer="sa_parks330")
parks660 = gpd.read_file(data, layer="sa_parks660")
parks1000 = gpd.read_file(data, layer="sa_parks1000")  

# service areas check
# check length
parks330.geometry.length.describe()
parks330[parks330.geometry.length == 0].count() # 26 parks with no service area
# technically not an issue, since e2sfca algorithm sorts this out (maybe not optimized)

# check areas
parks330['area'].describe()
parks330[parks330['area'] <= 30].count() # 87 parks below 30sqm
# very small parks (assume below 30sqm) are due to merging process errors

# filter out small parks (the merging process created some zero area entries)
parks330 = parks330.query("area >= 30")
parks660 = parks660.query("area >= 30")
parks1000 = parks1000.query("area >= 30")

# Census catchement areas
census = gpd.read_file(data, layer="census_centroids")
# census330 = gpd.read_file(data, layer = 'census330')
# census660 = gpd.read_file(data, layer = 'census660')
# census1000 = gpd.read_file(data, layer = 'census1000')

# fix census datatypes (they are almost all 'objects')
string_columns = ["KEY_CODE_3", "name_ja", "name_en"]
for col in census.columns:
    if col not in string_columns and col != "geometry":
        census[col] = pd.to_numeric(census[col], errors="coerce", downcast="integer")

#TODO apply accessibility function to different subset of parks (by size)
#TODO clean census units
# there are census units with very low population count
# low values in pop affect the ugs to population ratio



########################################################################
# E2SFCA  ##############################################################
########################################################################
# STEP 1: get UGS to population ratios ################################
########################################################################
accessibility_dict1 = get_accessibility_dict(accesses, parks330, parks660, parks1000, census)

# GET THE UGS TO POPULATION RATIO

ugs_pop_ratios = get_ugs_to_pop_ratios(accessibility_dict1, census)
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
for park_id, park_data in accessibility_dict1.items():
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
    if park_id == 5036: #highest ugs_to_pop ratio
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

# filter the "outliers"
o3000 = list(census[census['pop_tot']>3000]["KEY_CODE_3"].unique())
o4000 = list(census[census['pop_tot']>4000]["KEY_CODE_3"].unique())
u100 = list(census[census['pop_tot']<100]["KEY_CODE_3"].unique())
u50 = list(census[census['pop_tot']<50]["KEY_CODE_3"].unique())
u10 = list(census[census['pop_tot']<10]["KEY_CODE_3"].unique())

# The following assignment should be integrated in the function below
census_o3000 = census[census['KEY_CODE_3'].isin(o3000)]
census_o4000 = census[census['KEY_CODE_3'].isin(o4000)]
census_u100 = census[census['KEY_CODE_3'].isin(u100)]
census_u50 = census[census['KEY_CODE_3'].isin(u50)]
census_u10 = census[census["KEY_CODE_3"].isin(u10)]

def plot_census_points_with_basemap(
    gdf, 
    buffer_factor=1.5, 
    title="Filtered Census Points in Tokyo (23 Special Wards)", 
    color="red", 
    markersize=10, 
    alpha=0.7, 
    basemap_source=ctx.providers.CartoDB.Positron,
    zoom=10
):
    """
    Plot census points on a basemap with adjustable parameters.
    
    Parameters:
        gdf (GeoDataFrame): GeoDataFrame to plot (must include geometries).
        buffer_factor (float): Factor to expand map bounds for zooming out.
        title (str): Title for the plot.
        color (str): Color of the points.
        markersize (int): Size of the points.
        alpha (float): Transparency of the points.
        basemap_source: Contextily basemap source.
        zoom (int): Optional zoom level for the basemap.
    """
    gdf = gdf.to_crs(epsg=3857) # for baseline compatibility
    
    # get bounds so I can increase them when I have small area
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    # Expand the bounds to zoom out
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]

    expanded_bounds = [
        bounds[0] - x_range * (buffer_factor - 1),  # Min X
        bounds[1] - y_range * (buffer_factor - 1),  # Min Y
        bounds[2] + x_range * (buffer_factor - 1),  # Max X
        bounds[3] + y_range * (buffer_factor - 1),  # Max Y
    ]

    # Plot the census points
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.plot(ax=ax, color=color, markersize=markersize, alpha=alpha, label="Filtered Census Units")

    # Add the basemap
    ctx.add_basemap(ax, source=basemap_source, zoom=zoom)

    # Set the expanded bounds as limits
    ax.set_xlim(expanded_bounds[0], expanded_bounds[2])  # Set x-axis limits
    ax.set_ylim(expanded_bounds[1], expanded_bounds[3])  # Set y-axis limits

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.legend(loc="upper left")
    ax.axis("off")  # Turn off axis labels

    # Show the plot
    plt.show()
    
plot_census_points_with_basemap(census_o3000)
plot_census_points_with_basemap(census_o4000, buffer_factor=5)
plot_census_points_with_basemap(census_u100)
plot_census_points_with_basemap(census_u50)
plot_census_points_with_basemap(census_u10)