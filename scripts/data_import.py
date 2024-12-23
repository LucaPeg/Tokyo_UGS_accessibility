# NECESSARY LIBRARIES
import geopandas as gpd
import fiona

gpkg_path = "C:\\Users\\lucap\\Documents\\GitHub\\Tokyo_UGS_accessibility\\data\\final\\ugs_accessibility.gpkg"
layers = fiona.listlayers(gpkg_path)

# Load the layers
census_330 = gpd.read_file(gpkg_path, layer="census_330")
census_660 = gpd.read_file(gpkg_path, layer="census_660")
census_1000 = gpd.read_file(gpkg_path, layer="census_1000")

park_330 = gpd.read_file(gpkg_path, layer="park_330")
park_660 = gpd.read_file(gpkg_path, layer="park_660")
park_1000 = gpd.read_file(gpkg_path, layer="park_1000")

census_data = gpd.read_file(gpkg_path, layer="census_data")
centroids = gpd.read_file(gpkg_path, layer="relevant_snapped_census_centroids")
