import geopandas as gpd
import pandas as pd
from collections import defaultdict
import contextily as ctx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def get_accessibility_dict(accesses, parks_sa330, parks_sa660, parks_sa1000, census_gdf):
    """
    This function needs two arguments:
    accesses = geodf containing park accesses and related parks characteristics
    parks330 = smallest service area for parks
    parks660 = intermediate service area for parks
    parks1000 = largest service area for parks
    census = census centroids snapped to road network, containin census unit info
    """
    accessibility_dict = {}  # initialize empty dictionary
    assigned_centroids_per_park = defaultdict(set)
    park_areas_from_accesses = accesses.groupby("park_id")["area"].first().to_dict()
    census_gdf['buffered'] = census_gdf.geometry.buffer(1)
    for park_id in park_areas_from_accesses:
        accessibility_dict[park_id] = {
            "park_area": int(park_areas_from_accesses[park_id]),
            "330m": [],
            "660m": [],
            "1000m": [],
        }

    joined_330 = gpd.sjoin(parks_sa330, census_gdf.set_geometry("buffered"), how="inner", predicate="intersects")
    joined_660 = gpd.sjoin(parks_sa660, census_gdf.set_geometry("buffered"), how="inner", predicate="intersects")
    joined_1000 = gpd.sjoin(parks_sa1000, census_gdf.set_geometry("buffered"), how="inner", predicate="intersects")

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

    filtered_accessibility_dict = {
        park_id: data
        for park_id, data in accessibility_dict.items()
        if data["330m"] or data["660m"] or data["1000m"]
    }

    return filtered_accessibility_dict

def get_ugs_to_pop_ratios(accessibility, census_centroids, weights=None):
    """
    Starting from the accessibility dictionary, this function yields the ugs to population ratios
    Args:
        accessibility (dict): dictionary with the park area and the census units
            reachable from a given park, divided by travel time zone. This can be obtained
            by using the function "get_accessibility_dict"
        census: gdf containing the census centroids snapped to the road network.
        weights (dict): dictionary containing the weights for the travel time zones.
            default is 1, 0.44, 0.03
    """
    ugs_population_ratio = {}  # initialize empty dictionary
    population_col = (
        "pop_tot"  # name of the column indeintifying total population in census gdf
    )
    if weights is None:
        weights = {
            "330m": 1,
            "660m": 0.44,
            "1000m": 0.03,
        }  # default is Gaussian sharp decay

    # Loop through each park in the accessibility_dict to calculate the UGS-to-population ratio
    for park_id, park_data in accessibility.items():
        park_area = park_data.get("park_area")  # Area of the park
        # print(f"Processing park_id: {park_id}. Park area: {park_area}") #debug
        total_weighted_population = 0  # initialize weighted pop

        # Loop through each zone and calculate the weighted population
        for zone, centroids in park_data.items():
            if (
                zone in weights
            ):  # this is necessary as zone includes also the "Park area", it gets the park area value as respective centroid.
                weight = weights[zone]
                # For each census unit in this zone, sum the population and multiply by the zone's weight
                population_sum = 0  # reset the pop running sum each time
                for centroid in centroids:
                    population_sum += census_centroids[
                        census_centroids["KEY_CODE_3"] == centroid
                    ][population_col].sum()
                # print(f"Population in {zone}: {population_sum}") #debug
                # Multiply by the weight for this zone
                total_weighted_population += population_sum * weight
                total_weighted_population = int(total_weighted_population)
        # Compute the UGS-to-population ratio for this park
        if total_weighted_population > 0:
            ugs_population_ratio[park_id] = park_area / total_weighted_population
        else:
            ugs_population_ratio[park_id] = 0  # is it best to use 0 or None?

    return ugs_population_ratio

def plot_census_points_with_basemap(
    census_gdf,
    sign,
    threshold,
    buffer_factor=1.5,
    color="red",
    markersize=10,
    alpha=0.7,
    zoom=10,
    basemap_source=ctx.providers.CartoDB.Positron,
    title=None
):
    """
    Plot census points on a basemap with adjustable parameters.

    Parameters:
        census (GeoDataFrame): census gdf (must include geometries).
        sign (str): 'over' or 'under'
        threshold (int): population threshold to plot
        buffer_factor (float): Factor to expand map bounds for zooming out.
        title (str): Title for the plot.
        color (str): Color of the points.
        markersize (int): Size of the points.
        alpha (float): Transparency of the points.
        basemap_source: Contextily basemap source.
        zoom (int): Optional zoom level for the basemap.
    """
    if title is None:
        title = f"Census units {sign} {str(int(threshold))} inhabitants"
    if sign not in ["over", "under"]:
        raise ValueError("The sing must be either 'over' or 'under'")
    census = census_gdf.to_crs(epsg=3857)  # for baseline compatibility
    if sign == "over":
        filtered_units = census[census["pop_tot"] > threshold]["KEY_CODE_3"].unique()
    elif sign == "under":
        filtered_units = census[census["pop_tot"] < threshold]["KEY_CODE_3"].unique()

    filtered_units_list = list(filtered_units)
    gdf = census[census["KEY_CODE_3"].isin(filtered_units_list)]
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
    gdf.plot(
        ax=ax,
        color=color,
        markersize=markersize,
        alpha=alpha,
    )

    # Add the basemap
    ctx.add_basemap(ax, source=basemap_source, zoom=zoom)

    # Set the expanded bounds as limits
    ax.set_xlim(expanded_bounds[0], expanded_bounds[2])  # Set x-axis limits
    ax.set_ylim(expanded_bounds[1], expanded_bounds[3])  # Set y-axis limits

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.axis("off")  # Turn off axis labels

    # Show the plot
    plt.show()

def get_census_served(parks, accessibility): 
    """
    Given a list of parks and the accessibility dictionary, yields the respective census units
        Args:
        parks (list): list of the park_id of interests
        accessibilit (dict): accessibility dictionary obtained with the function above

    Returns:
        dict:{park_id : [list of census units]}
    """   
    tt_zones = ['330m', '660m','1000m']
    filt_accessibility = {key: accessibility[key] for key in parks if key in accessibility}
    park_service = {}
    for park_id, park_data in filt_accessibility.items():    
        serviced_centroids = []
        for key, data in park_data.items():
            if key in tt_zones:
                for centroid in data:
                    serviced_centroids.append(centroid)
            park_service[park_id] = serviced_centroids
    return park_service

def get_people_served(parks, accessibility, census):
    """Given a list of parks and the census geodataframe, yields for each park how many people it serves.

    Args:
        parks (list): list of park you want to check
        accessibility: accessibility dictionary
        census (_type_): census geodataframe, whose population columns is called "pop_tot"

    Returns:
        _dict_: {park_id : total_people_served}
    """
    park_service = get_census_served(parks, accessibility)
    people_served = {}
    for park_id, census_units in park_service.items():
        park_population = 0
        for census_unit in census_units:
            census_pop = int(census.loc[census["KEY_CODE_3"]==census_unit, 'pop_tot'].iloc[0])
            park_population += census_pop
        people_served[park_id] = park_population
    return people_served

def get_census_catchment(accesses, census330, census660, census1000, census):
    """Given the park accesses and the census units, returns the parks accessible from each census unit

    Args:
        accesses (_geodataframe_): park access dataframe containing access points and information on parks
        census330 (_geodataframe_): 330m service area around each census unit centroid 
        census660 (_geodataframe_): 660m service area around each census unit centroid
        census1000 (_geodataframe_): 1000m service area around each census unit centroid
        census (_geodataframe_): geodataframe of all the snapped centroids units
    Returns:
        dict: dictionary whose key is the census id. The values are dictionaries for the three travel time zones.
    """
    census_catchment = {}  # initialize empty dictionary
    assigned_park_per_census = defaultdict(set)
    for census_id in census['KEY_CODE_3']:
        census_catchment[census_id] = {
            "330m": [],
            "660m": [],
            "1000m": [],
        }
    accesses['buffered'] = accesses.geometry.buffer(1)
    joined_330 = gpd.sjoin(census330, accesses.set_geometry("buffered"), how="inner", predicate="intersects")
    joined_660 = gpd.sjoin(census660, accesses.set_geometry("buffered"), how="inner", predicate="intersects")
    joined_1000 = gpd.sjoin(census1000, accesses.set_geometry("buffered"), how="inner", predicate="intersects")

    for census_id, group in joined_330.groupby("KEY_CODE_3"):
        unique_parks = group["park_id"].unique()
        unassigned_parks = [
             int(c) for c in unique_parks if c not in assigned_park_per_census[census_id]
        ]
        if unassigned_parks:
            census_catchment[census_id]["330m"].extend(unassigned_parks)
            assigned_park_per_census[census_id].update(unassigned_parks)

    # Now process the 660m service area (excluding those already assigned in 330m)
    for census_id, group in joined_660.groupby("KEY_CODE_3"):
        unique_parks = group["park_id"].unique()
        unassigned_parks = [
             int(c) for c in unique_parks if c not in assigned_park_per_census[census_id]
        ]
        if unassigned_parks:
            census_catchment[census_id]["660m"].extend(unassigned_parks)
            assigned_park_per_census[census_id].update(unassigned_parks)


    # Finally, process the 1000m service area (excluding those already assigned in 330m and 660m)
    for census_id, group in joined_1000.groupby("KEY_CODE_3"):
        unique_parks = group["park_id"].unique()
        unassigned_parks = [
            int(c) for c in unique_parks if c not in assigned_park_per_census[census_id]
        ]
        if unassigned_parks:
            census_catchment[census_id]["1000m"].extend(unassigned_parks)
            assigned_park_per_census[census_id].update(unassigned_parks)


    filtered_census_catchment = {
        census_id: data 
        for census_id, data in census_catchment.items() 
        if data["330m"] or data["660m"] or data["1000m"]
    }

    return filtered_census_catchment

def get_accessibility_index(dict_census_to_parks, census_centroids, dict_parks_to_census, weights=None):
    ratios = get_ugs_to_pop_ratios(dict_parks_to_census, census_centroids)
    e2sfca = {}
    if weights is None:
        weights = {"330m":1, "660m":0.44, "1000m":0.03} # default is Guassian sharp decay
    for census_unit, grouped_parks in dict_census_to_parks.items():
        tot_accessibility = 0
        for zone, parks in grouped_parks.items():
            if zone in weights: # this checks that the keys align (330m, 660m, etc)
                weight = weights[zone]
                sum_accessibility = 0
                for park in parks:
                    sum_accessibility += ratios[park]
            tot_accessibility += sum_accessibility*weight
        if tot_accessibility > 0:
            e2sfca[census_unit] = tot_accessibility
        else:
            e2sfca[census_unit] = 0
    return e2sfca

def get_accessibility_index_log(dict_census_to_parks, census_centroids, dict_parks_to_census, weights=None):
    ratios = get_ugs_to_pop_ratios(dict_parks_to_census, census_centroids)

    for k in ratios:  # take log
        ratios[k] = np.log1p(ratios[k])

    e2sfca = {}
    
    if weights is None:
        weights = {"330m": 1, "660m": 0.44, "1000m": 0.03}  # Default is Gaussian sharp decay
    
    for census_unit, grouped_parks in dict_census_to_parks.items():
        tot_accessibility = 0  # reset for each census unit
        
        for zone, parks in grouped_parks.items():
            if zone in weights:  
                weight = weights[zone]
                sum_accessibility = sum(ratios.get(park, 0) for park in parks)
                tot_accessibility += sum_accessibility * weight
        if tot_accessibility > 0:
            e2sfca[census_unit] = tot_accessibility
        else:
            e2sfca[census_unit] = 0
            
    return e2sfca

def plot_parks_by_ratio(parks, ugs_ratio_threshold, perimeter=None, basemap=True):
    """
    Plots park polygons above a ugs_ratio threshold. Color represents 'ugs_ratio' attribute.

    Parameters:
    - parks (GeoDataFrame): The GeoDataFrame containing park polygons (with 'ugs_ratio' attribute).
    - ugs_ratio_threshold (float): Minimum value of 'ugs_ratio' to plot a park.
    - perimeter_gdf (GeoDataFrame): Optional, overlays a boundary (like the study area one)
    - basemap (bool): If True, adds a basemap (Carto).
    """
    # filter parks
    filtered_parks = parks[parks['ugs_ratio'] > ugs_ratio_threshold]
    
    # colormap (green to red)
    cmap = LinearSegmentedColormap.from_list('green_red', ['green', 'yellow', 'red'])
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    filtered_parks.plot(
        ax=ax,
        column='ugs_ratio',  # color by 'ugs_ratio'
        cmap=cmap,
        legend=True,
        legend_kwds={'label': "UGS Ratio", 'orientation': "vertical"}
    )
    # Perimeter (optional)
    if perimeter is not None:
        perimeter.geometry.plot(ax=ax, color='black', linewidth=0.5, label='Study Area Boundary')
    
    # Basemap is by default
    if basemap:
        ctx.add_basemap(
            ax,
            crs=filtered_parks.crs.to_string(),
            source=ctx.providers.CartoDB.Positron  # Use a terrain basemap
        )
    
    ax.set_title(f"Parks with UGS Ratio > {ugs_ratio_threshold}", fontsize=14)
    ax.set_axis_off()
    
    plt.show()
    
def plot_parks_ratio_people(parks, ugs_ratio_threshold, perimeter_gdf=None, basemap=True):
    """
    Plots park polygons above a ugs_ratio threshold and overlays the study area perimeter.

    Parameters:
    - parks (GeoDataFrame): The GeoDataFrame containing park polygons (with 'ugs_ratio' and "affluency").
    - ugs_ratio_threshold (float): Minimum value of 'ugs_ratio' to plot a park.
    - perimeter_gdf (GeoDataFrame): Optional GeoDataFrame of the study area perimeter.
    - perimeter_gdf (GeoDataFrame): Optional, overlays a boundary (like the study area one)
    - basemap (bool): If True, adds a basemap (Carto).
    """
    # Filter parks
    filtered_parks = parks[parks['ugs_ratio'] > ugs_ratio_threshold]
    
    # Colormap (green to red)
    cmap = LinearSegmentedColormap.from_list('red_green', ['red', 'yellow', 'green'])
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    filtered_parks.plot(
        ax=ax,
        column='affluency',  # color by 'affluency'
        cmap=cmap,
        legend=True,
        legend_kwds={'label': "People living within 1km of the park", 'orientation': "vertical"}
    )
    
    # Perimeter (optional)
    if perimeter_gdf is not None:
        perimeter_gdf.geometry.plot(ax=ax, color='black', linewidth=0.5, label='Study Area Boundary')

    
    # Basemap (default)
    if basemap:
        ctx.add_basemap(
            ax,
            crs=filtered_parks.crs.to_string(),
            source=ctx.providers.CartoDB.Positron
        )
    
    ax.set_title(f"Parks with UGS Ratio > {ugs_ratio_threshold}", fontsize=14)
    ax.set_axis_off()
    ax.legend()
    
    plt.show()