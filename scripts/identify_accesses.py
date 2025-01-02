#################################################################
#### THIS DIDN'T WORK, I JUST USED QGIS AND THEN FIXED ON PYTHON
#################################################################

# LIBRARIES

import geopandas as gpd
import warnings
import os
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree

# IMPORT LAYERS
data = os.path.join("..\\data\\final\\Tokyo_UGS_accessibility.gpkg")

ugs = gpd.read_parquet(data, layer='ugs')
roads = gpd.read_file(data, layer = 'road_network')
roads_tree = STRtree(roads["geometry"].values) # creates a spatial index

# check that park_id is unique and CRS match
assert ugs.shape[0] == ugs["park_id"].nunique(), "park_id is not unique" 
assert ugs.crs == roads.crs, "crs do not match"
# check for invalid geometries
invalid_geometries = ugs[~ugs.is_valid]

if not invalid_geometries.empty:
    warnings.warn(f"There are {len(invalid_geometries)} invalid geometries in the dataset.", UserWarning)

# add the perimeter to UGS
perim = ugs[ugs.is_valid].copy()
perim['perimeter'] = perim.geometry.apply(lambda geom: geom.exterior)
perim = perim.set_geometry('perimeter')

road_sindex = roads.sindex # create spatial index for efficiency

intersecting = perim[perim.geometry.intersects(roads)].union_all()
non_intersecting = perim[~perim.geometry.intersects(roads)]



def create_equidistant_points(geom, distance=150):
    points = []
    length = geom.length # get perimeter length
    for i in range(0, int(length), distance):
        point = geom.interpolate(i) # interpolate point at distance i
        points.append(point)
    return points

non_intersecting['access_points'] = non_intersecting.geometry.apply(create_equidistant_points)

####################################################################
#### ASSUME I FOUND THE ACCESSES AND STORED THEM INTO THE DATA FILE
#### for now the park accesses were found through QGIS
####################################################################


