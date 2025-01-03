import geopandas as gpd
import os

# IMPORT LAYERS
data = os.path.join("..\\data\\final\\Tokyo_UGS_accessibility.gpkg")

# Parks catchement areas
accesses = gpd.read_file(data,layer = 'cleaned_park_accesses)')
parks330 = gpd.read_file(data, layer = 'parks330')
parks660 = gpd.read_file(data, layer = 'parks660')
parks1000 = gpd.read_file(data, layer = 'parks1000')

# Census catchement areas
census_centroids = gpd.read_file(data, layer = 'census_centroids')
census330 = gpd.read_file(data, layer = 'census330')
census660 = gpd.read_file(data, layer = 'census660')
census1000 = gpd.read_file(data, layer = 'census1000')

# E2SFCA 
# Step 1: UGS to population ratios
# 