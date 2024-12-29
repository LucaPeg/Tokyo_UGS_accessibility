# LIBRARIES

import geopandas as gpd
import warnings
import os
from shapely.geometry import LineString

# IMPORT LAYERS
ugs_path = os.path.join("..\\data\\final\\ugs.parquet")
roads_path = os.path.join("..\\data\\final\\road_network.parquet")

ugs = gpd.read_parquet(ugs_path)
roads = gpd.read_parquet(roads_path)

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

# find the intersections
buffered_roads = roads.buffer(0.001)
intersecting = perim[perim.geometry.intersects(buffered_roads.union_all)]
non_intersecting = perim[~perim.geometry.intersects(buffered_roads.union_all)]
