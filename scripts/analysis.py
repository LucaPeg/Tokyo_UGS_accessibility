# Import libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import fiona
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from accessibility_functions import (plot_census_points_with_basemap,
                                     get_census_served,
                                     get_people_served,
                                     plot_parks_ratio_people,
                                     plot_parks_by_ratio )


# import data
data = os.path.join("..\\data\\final\\analysis_data.gpkg")
tokyo_wards = gpd.read_file(data, layer='wards')
census = gpd.read_file(data, layer='census_for_regression')

# ADDING EDUCATIONAL ATTAINEMENT DATA

# education_data_path = os.path.join('..\\data\\provisional\\final_work_census250.xlsx')
# education_data = pd.read_excel(education_data_path)
# education_data.replace('*', np.nan, inplace=True)  # japanese census has * instead of NULL
# # merging process
# education_data['KEY_CODE'] = education_data['KEY_CODE'].astype(int) 
# census['KEY_CODE_3'] = census['KEY_CODE_3'].astype(int)
# census = census.merge(education_data, left_on='KEY_CODE_3', right_on='KEY_CODE', how='left')
# # fix columns
# census.drop(columns=["KEY_CODE"], inplace=True)
# non_numeric= ["name_ja", "name_en", "geometry"]
# for col in census.columns:
#     if col not in non_numeric:
#         census[col] = pd.to_numeric(census[col], errors="coerce")  # Convert non-numeric to NaN
# census.to_file(data, layer='census_for_regression')
# fiona.listlayers(data)
# census = gpd.read_file(data,layer='census_for_regression')


# SELECTING / CREATING VARIABLES FOR REGRESSION ANALYSIS
census['prop_young_pop'] = census['pop_under14']/census['pop_tot']
census['prop_o65_pop'] = census['pop_over65']/census['pop_tot']
census['prop_o75_pop'] = census['pop_over75']/census['pop_tot']
census['prop_foreign_pop'] = census['pop_foreign']/census['pop_tot']
census['prop_hh_o65'] = census['househ_w_over65yo'] / census['n_households']
census['hh_only_elderly'] = census['singleelderly_househ'] + census['couplelederly_househ']
census['prop_hh_only_elderly'] = census['hh_only_elderly'] / census['n_households']
census['prop_managers'] = census['managers'] / census['total_workers']
census['n_managers_professionals'] = census['managers'] + census['professional_workers']
census['prop_high_earning'] = census['n_managers_professionals'] / census['total_workers']
census['prop_uni_graduates'] = census['grad_university']/census['tot_graduates'] 
census['prop_15_65'] =
census['prop_1hh'] = 
census['prop_2hh'] = 
census['prop_3hh'] = 
census['prop_4hh'] =
census['prop_5hh'] = 
census['prop_6hh'] =
census['prop_7hh'] =
census['prop_low_wage'] 


# EXPLORATORY DATA ANALYSIS
# check issue in population distributions: based on results fixe the e2sfca.py
pd.set_option('display.max_columns', None)
print(census.isnull().sum().to_string) # issue only for 13 census units for high earning and 5 for uni graduates
census.describe()
census['pop_tot'].quantile(0.05) #234

# some proportions have max values above 1: PROBLEM
sns.boxplot(census['prop_o65_pop'])
sns.boxplot(census['prop_o75_pop']) # this proportion is 1
sns.boxplot(census['prop_foreign_pop']) # this proportion is 2
sns.boxplot(census['prop_hh_only_elderly'])
sns.boxplot(census['prop_managers']) # this is 1
sns.boxplot(census['prop_high_earning']) #  1
sns.boxplot(census['prop_uni_graduates']) #  1


pd.set_option('display.max_columns', None)
problematic_census = []
census[census['prop_o65_pop'] > 0.8][['KEY_CODE_3','pop_tot','pop_over65']]
census[census['prop_o75_pop'] > 0.7][['KEY_CODE_3','pop_tot','pop_over75']]
census[census['prop_foreign_pop'] > 0.8][['KEY_CODE_3','pop_tot','pop_foreign']]
census[census['prop_hh_only_elderly'] > 0.8][['KEY_CODE_3','pop_tot','n_households','singleelderly_househ','couplelederly_househ']]
census[census['prop_managers'] > 0.8][['KEY_CODE_3','total_workers','pop_tot','managers','professional_workers']]
census[census['prop_high_earning'] > 0.8][['KEY_CODE_3','total_workers','professional_workers', 'prop_high_earning']]
census[census['prop_uni_graduates'] > 0.8]["KEY_CODE_3"]

problematic_census = set()
problematic_census.update(census[census['prop_o65_pop'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_o75_pop'] > 0.7]["KEY_CODE_3"])
problematic_census.update(census[census['prop_foreign_pop'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_managers'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_hh_only_elderly'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_high_earning'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_uni_graduates'] > 0.8]["KEY_CODE_3"])

census = census[~census['KEY_CODE_3'].isin(problematic_census)]

census[census['price_mean']>10000000]

# census with high population
plot_census_points_with_basemap(census, 'over', 4000) # do these locations make sense?
census[census['pop_tot']>4000]

# indepedent variables
X = ['pop_tot', '', '','prop_young_pop', 'prop_o65_pop', 'prop_o75_pop', 'prop_foreign_pop',
       'prop_hh_only_elderly', 'prop_managers',
       'prop_high_earning', 'prop_uni_graduates', 'price_mean',
       ]
relevant_attributes = ['prop_young_pop', 'prop_o65_pop', 'prop_o75_pop', 'prop_foreign_pop',
       'prop_hh_only_elderly', 'prop_managers',
       'prop_high_earning', 'prop_uni_graduates', 'price_mean',
       'full_ugs_accessibility', 'vl_ugs_accessibility',
       'lg_ugs_accessibility', 'md_ugs_accessibility', 'sm_ugs_accessibility']

rdata = census[relevant_attributes]

# distribution
# Set up the figure
plt.figure(figsize=(15, 10))

# Plot histograms for all numerical columns
for i, column in enumerate(rdata.columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(rdata[column], kde=True, bins=30)
    plt.title(column)
    plt.tight_layout()

plt.show()

# Calculate correlation matrix
corr_matrix = rdata.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Pairplot for selected columns
#sns.pairplot(rdata)
#plt.show()

# Set up the figure
plt.figure(figsize=(15, 10))

# Plot boxplots for all numerical columns
for i, column in enumerate(rdata.columns, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y=rdata[column])
    plt.title(column)
    plt.tight_layout()

plt.show()

# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data["Variable"] = rdata.columns
vif_data["VIF"] = [variance_inflation_factor(rdata.values, i) for i in range(rdata.shape[1])]

# Check for missing values
print(rdata.isnull().sum())
rdata = rdata.replace(np.nan).dropna()

vif_data = pd.DataFrame()
vif_data["Variable"] = rdata.columns
vif_data["VIF"] = [variance_inflation_factor(rdata.values, i) for i in range(rdata.shape[1])]

print(vif_data) # obviously



## PCA ################################################################## 

# remove non numeric column and target variables
features = rdata.drop(columns=[ #'KEY_CODE_3', 'name_ja', 'name_en', 'geometry', 
                                'full_ugs_accessibility', 'vl_ugs_accessibility',
                                'lg_ugs_accessibility','md_ugs_accessibility','sm_ugs_accessibility'])

# data standardization (PCA is influenced by measurement unit)
scaler = StandardScaler()
rdata_standardized = scaler.fit_transform(features)

# actual PCA computation
pca = PCA()
pca.fit(rdata_standardized)

# check explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6)) # scree plot
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Scree Plot')
plt.show()

# select number of components according to the scree plot
n_components = #(cumulative_explained_variance <= 0.75).sum()
pca = PCA(n_components=n_components)
rdata_pca = pca.fit_transform(rdata_standardized)

# create a df
rdata_pca_df = pd.DataFrame(rdata_pca, columns=[f"PC{i+1}" for i in range(n_components)])
print(rdata_pca_df.head())

# look at loadings to interpret the components
loadings = pd.DataFrame(pca.components_, columns=features.columns, index=[f"PC{i+1}" for i in range(n_components)])
print(loadings)

# TODO PCA with all features found in Census.

# TODO Gini index of accessibility
# TODO Lorenz curve of accessibility
# TODO GWR: fix issue with library  