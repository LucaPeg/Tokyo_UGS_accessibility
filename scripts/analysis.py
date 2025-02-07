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
data = os.path.join("..\\data\\final\\ugs_analysis_data.gpkg")
tokyo_wards = gpd.read_file(data, layer='wards')
census = gpd.read_file(data, layer='census_for_analysis')

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
# census.to_file(data, layer='census_for_analysis')
# fiona.listlayers(data)
# census = gpd.read_file(data,layer='census_for_analysis')

# SELECTING / CREATING VARIABLES FOR REGRESSION ANALYSIS ########################################
lower_wage = ['cleaning_workers', 'construction_workers', 'transport_machinery_workers',
    'production_process_workers', 'agri_workers', 'service_workers' ]
high_wage = ['managers','professional_workers']

census['prop_young_pop'] = census['pop_under14']/census['pop_tot']
census['prop_o65_pop'] = census['pop_over65']/census['pop_tot']
census['prop_o75_pop'] = census['pop_over75']/census['pop_tot']
census['prop_foreign_pop'] = census['pop_foreign']/census['pop_tot']
census['prop_hh_o65'] = census['househ_w_over65yo'] / census['n_households']
census['hh_only_elderly'] = census['singleelderly_househ'] + census['couplelederly_househ']
census['prop_hh_only_elderly'] = census['hh_only_elderly'] / census['n_households']
census['prop_low_wage'] = census[lower_wage].sum(axis=1)/census['total_workers']
census['prop_high_wage'] = census[high_wage].sum(axis=1)/census['total_workers']
census['prop_managers'] = census['managers'] / census['total_workers']
census['prop_uni_graduates'] = census['grad_university']/census['tot_graduates'] 
census['prop_15_64'] = census['pop_15_64']/census['pop_tot']
census['prop_1hh'] = census['1p_househ']/census['n_households']
census['prop_2hh'] = census['2p_househ']/census['n_households']
census['prop_3hh'] = census['3p_househ']/census['n_households']
census['prop_4hh'] = census['4p_househ']/census['n_households']
census['prop_5hh'] = census['5p_househ']/census['n_households']
census['prop_6hh'] = census['6p_househ']/census['n_households']
census['prop_7hh'] = census['7p_househ']/census['n_households']
census['prop_hh_u6'] = census['househ_w_under6yo']/census['n_households']
census['prop_hh_head20'] = census['headhouseh_in20s']/census['n_households']

# INEQUALITY ANALYSIS ##########################################################################

accessibility_measures = [
    'log_vl_ugs_accessibility', 'log_lg_ugs_accessibility',
    'log_md_ugs_accessibility', 'log_sm_ugs_accessibility',
    'log_full_ugs_accessibility'
]

for measure in accessibility_measures:  # normalize accessibility indexes
    min_val = census[measure].min()
    max_val = census[measure].max()
    census[measure + '_norm'] = (census[measure] - min_val) / (max_val - min_val)
    
def lorenz_curve(data):
    """Compute the Lorenz curve."""
    data_sorted = np.sort(data)
    cumulative = np.cumsum(data_sorted) / data_sorted.sum()
    cumulative = np.insert(cumulative, 0, 0)  # Add (0,0) to the curve
    return cumulative



# LORENZ CURVE PLOT
plt.figure(figsize=(10,8))

for measure in accessibility_measures:
    data =census[measure].values
    lorenz = lorenz_curve(data)
    plt.plot(np.linspace(0, 1, len(lorenz)), lorenz, label=measure)

# equality line
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Equality Line')

plt.xlabel('Cumulative Share of Population')
plt.ylabel('Cumulative Share of Accessibility')
plt.title('Lorenz Curves for Accessibility Measures')
plt.legend()
plt.grid(True)
plt.show()

# GINI INDEX
def gini_coefficient(data):
    """Compute the Gini coefficient."""
    data_sorted = np.sort(data)
    n = len(data)
    cumulative = np.cumsum(data_sorted) / data_sorted.sum()
    gini_index = 1 - 2 * np.trapz(cumulative, dx=1/n)
    return gini_index

# compute Gini index for each accessibility measure
for measure in accessibility_measures:
    data = census[measure + '_norm'].dropna().values
    gini = gini_coefficient(data)
    print(f'Gini Coefficient for {measure}: {gini:.2f}')
    
#####################################################################################
# REGRESSION ANALYSIS ###############################################################
#####################################################################################

# TODO add dummy for 'close to riverside'
# TODO add dummy for 'harbour area' / remove observations in harbour area 
# harbour area observations may potentially be removed before computing Gini/Lorenz

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
sns.boxplot(census['prop_15_64']) 
sns.boxplot(census['prop_1hh']) 
sns.boxplot(census['prop_hh_head20']) 


pd.set_option('display.max_columns', None)
census[census['prop_o65_pop'] > 0.8][['KEY_CODE_3','pop_tot','pop_over65']]
census[census['prop_o75_pop'] > 0.7][['KEY_CODE_3','pop_tot','pop_over75']]
census[census['prop_foreign_pop'] > 0.8][['KEY_CODE_3','pop_tot','pop_foreign']]
census[census['prop_hh_only_elderly'] > 0.8][['KEY_CODE_3','pop_tot','n_households','singleelderly_househ','couplelederly_househ']]
census[census['prop_managers'] > 0.8][['KEY_CODE_3','total_workers','pop_tot','managers','professional_workers']]
census[census['prop_high_earning'] > 0.8][['KEY_CODE_3','total_workers','professional_workers', 'prop_high_earning']]
census[census['prop_uni_graduates'] > 0.8]["KEY_CODE_3"]

census[census['prop_15_64'] > 0.9]["KEY_CODE_3"].count()
census[census['prop_1hh'] > 0.9]["KEY_CODE_3"].count()
census[census['prop_hh_head20'] > 0.8]["KEY_CODE_3"].count()


problematic_census = set()
problematic_census.update(census[census['prop_o65_pop'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_o75_pop'] > 0.7]["KEY_CODE_3"])
problematic_census.update(census[census['prop_foreign_pop'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_managers'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_hh_only_elderly'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_high_wage'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_uni_graduates'] > 0.8]["KEY_CODE_3"])
problematic_census.update(census[census['prop_15_64'] > 0.95]["KEY_CODE_3"])
problematic_census.update(census[census['prop_1hh'] > 0.95]["KEY_CODE_3"])
problematic_census.update(census[census['prop_hh_head20'] > 0.8]["KEY_CODE_3"])

census = census[~census['KEY_CODE_3'].isin(problematic_census)]

census.describe()
# census with high population
plot_census_points_with_basemap(census, 'over', 4000) # do these locations make sense?
census[census['pop_tot']>4000]

# indepedent variables
relevant_attributes = ['prop_young_pop', 'prop_o65_pop', 'prop_o75_pop', 'prop_foreign_pop',
       'prop_hh_only_elderly', 'prop_managers', 'prop_high_wage', 'prop_low_wage', 
       'prop_uni_graduates', 'price_mean', 'prop_15_64', 'prop_1hh', 'prop_hh_head20',
       'log_vl_ugs_accessibility', 'log_vl_ugs_accessibility_norm',
       'log_full_ugs_accessibility', 'log_full_ugs_accessibility_norm' ]

rdata = census[relevant_attributes]

# analayze the distirbution of the regressors
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


# Check for missing values
rdata = rdata.replace(np.nan, 0)

vif_data = pd.DataFrame()
vif_data["Variable"] = rdata.columns
vif_data["VIF"] = [variance_inflation_factor(rdata.values, i) for i in range(rdata.shape[1])]

print(vif_data) # obviously crazy high 


## PCA ################################################################## 

# remove non numeric column and target variables
features = rdata.drop(columns=[ #'KEY_CODE_3', 'name_ja', 'name_en', 'geometry', 'full_ugs_accessibility', 'lg_ugs_accessibility','md_ugs_accessibility','sm_ugs_accessibility'
                                 'log_vl_ugs_accessibility', 'log_full_ugs_accessibility',
                                 'log_vl_ugs_accessibility_norm', 'log_full_ugs_accessibility_norm'
                                ])


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
n_components = 3#(cumulative_explained_variance <= 0.75).sum()
pca = PCA(n_components=n_components)
rdata_pca = pca.fit_transform(rdata_standardized)
rdata_pca_df = pd.DataFrame(rdata_pca, columns=[f"PC{i+1}" for i in range(n_components)])

# look at loadings to interpret the components
loadings = pd.DataFrame(pca.components_, columns=features.columns, index=[f"PC{i+1}" for i in range(n_components)])
print(loadings)


# inspect correlation between components and accessibility

rdata_pca_df["log_vl_ugs_accessibility"] = rdata["log_vl_ugs_accessibility"].values
correlation_matrix = rdata_pca_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation between PCA Components and vl_ugs_accessibility")
plt.show()

# TODO associate ward names to parks using wards layer. Use it for visualizations

