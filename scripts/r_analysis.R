
library('tidyverse')
library('janitor')
library('spdep')
library('tmap')
library('GWmodel')
library('dplyr')
library('sf')
library('ggplot2')





Sys.setlocale("LC_ALL", "English_United States.1252")
Sys.setlocale("LC_ALL", "English")

setwd("C:/Users/Luca/Documents/Tokyo_UGS_accessibility/")

gdf <- st_read("data/final/regression_data")
wards <- st_read("data/final/ugs_analysis_data.gpkg", layer="wards")
st_crs(gdf) == st_crs(wards) # Check CRS
data <- st_join(gdf, wards, join = st_within)

## EDA #########################################################################

variables <- c('prop_young_pop', 'prop_o65_pop', 'prop_o75_pop', 'prop_foreign_pop',
          'prop_hh_only_elderly', 'prop_managers', 'prop_high_wage', 'prop_low_wage', 
          'prop_uni_graduates', 'price_std', 'prop_15_64', 'prop_1hh', 'prop_hh_head20',
          'log_vl_ugs_accessibility_norm', 'log_full_ugs_accessibility_norm')

ward_medians <- data %>%
  group_by(name_en) %>%
  summarize(across(c('prop_young_pop', 'prop_hh_only_elderly','price_std',
                     'prop_managers','prop_high_wage',
                     'log_vl_ugs_accessibility_norm',
                     'log_full_ugs_accessibility_norm'), ~ median(.x, na.rm = TRUE)))

wards_merged <- wards %>%
  st_join(ward_medians, by='name_en')

# median high wage
tmap_mode("plot")
tm_shape(wards_merged) +
  tm_polygons("prop_high_wage", palette = "RdYlGn", 
              title = "Median Value", 
              style = "quantile") +
  tm_layout(frame = FALSE)

# Full accessibility
tmap_mode("plot")
tm_shape(wards_merged) +
  tm_polygons("log_full_ugs_accessibility_norm", palette = "RdYlGn", 
              title = "Median Value", 
              style = "quantile") +
  tm_layout(frame = FALSE)

# Accessibility to very large parks
tmap_mode("plot")
tm_shape(wards_merged) +
  tm_polygons("log_vl_ugs_accessibility_norm", palette = "RdYlGn", 
              title = "Median Value", 
              style = "quantile") +
  tm_layout(frame = FALSE)


## TESTING FOR SPATIAL AUTOCORRELATION ##
rdata <- data[c('prop_young_pop', 'prop_o65_pop', 'prop_o75_pop', 'prop_foreign_pop',
          'prop_hh_only_elderly', 'prop_managers', 'prop_high_wage', 'prop_low_wage', 
          'prop_uni_graduates', 'price_std', 'prop_15_64', 'prop_1hh', 'prop_hh_head20',
          'log_full_ugs_accessibility_norm','geometry')]

# Find k-nearest neighbors (e.g., k=3)
coords <- st_coordinates(st_centroid(rdata))
knn_nb <- knearneigh(coords, k = 8)
nb_knn <- knn2nb(knn_nb)
lw <- nb2listw(nb_knn, style = 'W')

moran.test(rdata$log_full_ugs_accessibility_norm, lw)


# USE MY ACTUAL MODELS # Lasso is apparently not viable
## REDUCED MODEL ###############################################################


summary(red_model)
# EVALUATE RESIDUALS ###########################################################


residuals <- gwr_model$SDF$residual
nb <- knn2nb(knearneigh(coords, k = 8))  # 5 nearest neighbors
w <- nb2listw(nb)
moran.test(residuals, w) # test spatial autocorrelation

plot(gwr_model$SDF$residual, main = "Residuals")
qqnorm(gwr_model$SDF$residual, main = "QQ Plot of Residuals")

# robust regression
r_reduced <- gwr.robust(
  formula_selected,
  data = rdata,
  bw = 2000,
  kernel = "bisquare",
  adaptive = TRUE
)
summary(r_reduced)

# mixed regression 
mixed_gwr <- gwr.mixed(
  formula_selected,
  data = rdata,
  bw = 2000,  
  kernel = "adaptive",
  fixed.vars = c("price_std"),  # Variables with global coefficients
  adaptive = TRUE
)
summary(mixed_gwr)

# SWITCH to point geometry for efficiency
gdf_centroids <- st_drop_geometry(gdf) %>%  # Remove polygon geometry
  st_as_sf(coords = c("longitude", "latitude"), crs = 32654)  # Replace with your column names
gdf_sp <- as(gdf_centroids, "Spatial")

# compute distance matrix
dMatrix <- gw.dist(coordinates(gdf_sp), longlat = TRUE)
# REDUCED MODEL ######################
formula_selected <- log_full_ugs_accessibility_norm ~ prop_foreign_pop + prop_young_pop+ prop_hh_only_elderly + prop_managers + price_std 

red_model <- ggwr.basic(
  formula_selected,
  data = gdf_sp,
  dMat = dMatrix,
  bw = 60,  
  kernel = "bisquare",
  adaptive = TRUE
)