import geopandas as gpd
import pandas as pd
from collections import defaultdict
import contextily as ctx
import matplotlib.pyplot as plt
import os


def get_accessibility_dict(accesses, parks330, parks660, parks1000, census):
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
    for park_id in park_areas_from_accesses:
        accessibility_dict[park_id] = {
            "park_area": int(park_areas_from_accesses[park_id]),
            "330m": [],
            "660m": [],
            "1000m": [],
        }

    joined_330 = gpd.sjoin(parks330, census, how="inner", predicate="intersects")
    joined_660 = gpd.sjoin(parks660, census, how="inner", predicate="intersects")
    joined_1000 = gpd.sjoin(parks1000, census, how="inner", predicate="intersects")

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
    title="Filtered Census Points",
    color="red",
    markersize=10,
    alpha=0.7,
    basemap_source=ctx.providers.CartoDB.Positron,
    zoom=10,
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
        label=f"Census units {sign} {threshold} inhabitants",
    )

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
