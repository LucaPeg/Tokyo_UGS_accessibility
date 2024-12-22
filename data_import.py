# NECESSARY LIBRARIES
import os
import geopandas as gpd
import pandas as pd
import numpy as np


# DATA IMPORT
# define directories
current_dir = os.getcwd()
data_folder = os.path.join(current_dir, "data\\final")
census_path = os.path.join(data_folder, "census_data.gpkg")
service_area_path = os.path.join(data_folder, "service_areas.gpkg")
# actual import
census_data = gpd.read_file(census_path)
parks330 = gpd.read_file(service_area_path, layer="park_330")
parks660 = gpd.read_file(service_area_path, layer="park_660")
parks1000 = gpd.read_file(service_area_path, layer="park_1000")
census330 = gpd.read_file(service_area_path, layer="census_330")
census660 = gpd.read_file(service_area_path, layer="census_660")
census1000 = gpd.read_file(service_area_path, layer="census_1000")
