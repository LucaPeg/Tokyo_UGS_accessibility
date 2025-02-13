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
np.float = float  # temporary fix for deprecated np.float 
import mgwr # to import this I changed the code of spatial/Lib/site-packages/libpysal/cg/kdtree.py
# change import statement to "from scipy import inf" to "from numpy import inf"

from mgwr.gwr import GWR
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

census['log_price'] = np.log(census['price_mean'])

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
# EXPLORATORY DATA ANALYSIS #########################################################

# TODO associate ward names to parks using wards layer. Use it for visualizations

# check issue in population distributions: based on results fixe the e2sfca.py
pd.set_option('display.max_columns', None)
census.describe()

# some proportions have max values above 1: PROBLEM
sns.boxplot(census['prop_o65_pop'])
sns.boxplot(census['prop_o75_pop']) # this proportion is 1
sns.boxplot(census['prop_foreign_pop']) # this proportion is 2
sns.boxplot(census['prop_hh_only_elderly'])
sns.boxplot(census['prop_managers']) # this is 1
sns.boxplot(census['prop_high_wage']) #  1
sns.boxplot(census['prop_uni_graduates']) #  1
sns.boxplot(census['prop_15_64']) 
sns.boxplot(census['prop_1hh']) 
sns.boxplot(census['prop_hh_head20']) 
sns.boxplot(census['log_price'])

pd.set_option('display.max_columns', None)
census[census['prop_o65_pop'] > 0.90][['KEY_CODE_3','pop_tot','pop_over65']]
census[census['prop_o75_pop'] > 0.7][['KEY_CODE_3','pop_tot','pop_over75']]
census[census['prop_foreign_pop'] > 0.8][['KEY_CODE_3','pop_tot','pop_foreign']]
census[census['prop_hh_only_elderly'] > 0.8][['KEY_CODE_3','pop_tot','n_households','singleelderly_househ','couplelederly_househ']]
census[census['prop_managers'] > 0.8][['KEY_CODE_3','total_workers','pop_tot','managers','professional_workers']]
census[census['prop_high_wage'] > 0.8][['KEY_CODE_3','total_workers','professional_workers', 'prop_high_wage']]
census[census['prop_uni_graduates'] > 0.8]["KEY_CODE_3"]
census[census['prop_15_64'] > 0.9]["KEY_CODE_3"].count()
census[census['prop_1hh'] > 0.9]["KEY_CODE_3"].count()
census[census['prop_hh_head20'] > 0.8]["KEY_CODE_3"].count()


problematic_census = set()
proportions = ['prop_o65_pop','prop_o75_pop', 'prop_foreign_pop','prop_managers',
               'prop_hh_only_elderly', 'prop_high_wage','prop_uni_graduates',
               'prop_15_64','prop_1hh','prop_hh_head20']

for attribute in proportions:
    problematic_census.update(census[census[attribute]>1]["KEY_CODE_3"])

census = census[~census['KEY_CODE_3'].isin(problematic_census)]

# census with high population
plot_census_points_with_basemap(census, 'over', 4000) # do these locations make sense?
census[census['pop_tot']>4000] # They do make sense -> Japanese Tower Buldings

# focus on the indepedent variables
relevant_attributes = ['prop_young_pop', 'prop_o65_pop', 'prop_o75_pop', 'prop_foreign_pop',
       'prop_hh_only_elderly', 'prop_managers', 'prop_high_wage', 'prop_low_wage', 
       'prop_uni_graduates', 'log_price', 'prop_15_64', 'prop_1hh', 'prop_hh_head20',
        'log_vl_ugs_accessibility_norm', 'log_full_ugs_accessibility_norm', 'KEY_CODE_3', 'geometry',
        ]


rdata = census[relevant_attributes]
rdata.loc[:,"KEY_CODE_3"] = rdata["KEY_CODE_3"].apply(str)
rdata.isna().sum()
rdata = rdata.replace(np.nan, 0)

# select numeric columns for PCA 
numeric_cols = rdata.select_dtypes(include=['number']).columns
filtered_data = rdata[numeric_cols]

# analyze regressors' distributions
plt.figure(figsize=(15, 10))
for i, column in enumerate(filtered_data.columns, 1):
    rows = int(np.ceil(len(filtered_data.columns) / 4))  # Dynamically calculate the number of rows
    plt.subplot(rows, 4, i)
    sns.histplot(filtered_data[column], kde=True, bins=30)
    plt.title(column)
    plt.tight_layout()
plt.show()

# Calculate correlation matrix
corr_matrix = filtered_data.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Plot boxplots for all numerical columns
plt.figure(figsize=(12, 8))
for i, column in enumerate(filtered_data.columns, 1):
    rows = int(np.ceil(len(filtered_data.columns) / 4))  # Dynamically calculate the number of rows
    plt.subplot(rows, 4, i)
    sns.boxplot(y=filtered_data[column])
    plt.title(column)
    plt.tight_layout()
plt.show()

## PCA ##########################################################################
# remove non numeric column and target variables
features = filtered_data.drop(columns=[ 
                                'log_vl_ugs_accessibility_norm', 'log_full_ugs_accessibility_norm'
                                ])
nonnumeric_cols = rdata[['KEY_CODE_3','geometry']]

features.isna().sum()
features.shape

nonnumeric_cols.isna().sum()
nonnumeric_cols.shape
                                                   
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
n_components = 3 # (cumulative_explained_variance <= 0.75).sum()
pca = PCA(n_components=n_components)
rdata_pca = pca.fit_transform(rdata_standardized)
rdata_pca_df = pd.DataFrame(rdata_pca, columns=[f"PC{i+1}" for i in range(n_components)])


pd.set_option('display.max_rows',None)
# PCA().fit_transform() does not retain the original index -> reset to rdata's
rdata_pca_df.index = rdata.index
# add PCs to rdata
rdata = pd.concat([rdata, rdata_pca_df], axis=1) #rdata and rdata_pca_df were both 7126, after this rdata becomes 7315

# look at loadings to interpret the components
loadings = pd.DataFrame(pca.components_, columns=features.columns, index=[f"PC{i+1}" for i in range(n_components)])
print(loadings)

# PCA VISUALIZATION ###

# Plot the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(rdata_pca_df['PC1'], rdata_pca_df['PC2'], alpha=0.2)
plt.title('PCA: PC1 vs PC2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Plot the first three principal components in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rdata_pca_df['PC1'], rdata_pca_df['PC2'], rdata_pca_df['PC3'], alpha=0.2)

ax.set_title('PCA: PC1 vs PC2 vs PC3')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.show()


# inspect correlation between components and accessibility

rdata_pca_df["log_vl_ugs_accessibility_norm"] = rdata["log_vl_ugs_accessibility_norm"].values
correlation_matrix = rdata_pca_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation between PCA Components and vl_ugs_accessibility")
plt.show()

#######################################################################################
# REGRESSION ANALYSIS #################################################################
#######################################################################################

# add dummy variables
dummies = census[['landfill','arakawa']]
rdata = pd.concat([rdata, dummies], axis=1)
# extract coordinates
rdata['centroid'] = rdata.geometry.centroid
rdata['longitude'] = rdata['centroid'].x
rdata['latitude'] = rdata['centroid'].y
coords = list(zip(rdata['longitude'], rdata['latitude']))


# BASIC GWR REGRESSION ############################################################################
X = rdata[['prop_young_pop', 'prop_o65_pop', 'prop_foreign_pop', 'prop_hh_only_elderly','prop_managers',
           'prop_low_wage','prop_uni_graduates','log_price']].values
y = rdata['log_vl_ugs_accessibility_norm'].values.reshape(-1, 1) # necessary for gwr (othersize size is [7126,])
bw = mgwr.sel_bw.Sel_BW(coords, y, X).search() # optimal bandwith selection
# model fitting
gwr_model = GWR(coords, y, X, bw)
gwr_results = gwr_model.fit()

gwr_results.summary()

#   base model diagnostics
filter_tc = gwr_results.filter_tvals()
filter_t = gwr_results.filter_tvals(alpha = 0.05)
rdata['fb'] = gwr_results.params[:,1] #
rdata['fb_tc'] = filter_tc[:,1]

fig, ax = plt.subplots(1, 3, figsize = (12, 3))
rdata.plot('fb',
            **{'edgecolor': 'black',
            'alpha': .65,
            'linewidth': .5},
        ax = ax[0],
        legend=True)
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].set_title('Parameter estimates')
rdata.plot('fb',
            **{'edgecolor': 'black',
            'alpha': .65,
            'linewidth': .5},
        ax = ax[1],
        legend=True)
rdata[filter_tc[:, 1] == 0].plot(color = 'grey',
                            ax = ax[1],
                            **{'edgecolor': 'black',
                            'linewidth': .5})
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title('Composite')
rdata.plot('fb',
            **{'edgecolor': 'black',
            'alpha': .65,
            'linewidth': .5},
        ax = ax[2],
        legend=True)
rdata[filter_tc[:, 1] == 0].plot(color = 'grey',
                            ax = ax[2],
                            **{'edgecolor': 'black',
                                'linewidth': .5})
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title('Composite with correction')
plt.savefig('testing')
plt.show()







# CHECK VIF
X_df = pd.DataFrame(X, columns=['prop_young_pop', 'prop_o65_pop', 'prop_foreign_pop', 'prop_hh_only_elderly','prop_managers',
           'prop_low_wage','prop_uni_graduates','log_price', ])  

# Compute VIF for each variable
vif_data = pd.DataFrame()
vif_data["Feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
print(vif_data)

# REDUCING MULTICOLLINEARITY ######################################################################
# PCA #############################################################################################
Xpca = rdata[['PC1', 'PC2', 'PC3']].values
bw_pca = mgwr.sel_bw.Sel_BW(coords, y, Xpca).search()
gwr_pca_model = GWR(coords, y, Xpca, bw_pca)
gwr_pca_results = gwr_pca_model.fit()
gwr_pca_results.summary()

# CHECK VIF with PCA
X_pca_df = pd.DataFrame(Xpca, columns=['PC1','PC2','PC3'])
vif_pca =  pd.DataFrame()
vif_pca["Feature"] = X_pca_df.columns
vif_pca["VIF"] = [variance_inflation_factor(X_pca_df.values, i) for i in range(X_pca_df.shape[1])]
print(vif_pca)

# MODEL SELECTION #################################################################################
X = rdata[['prop_foreign_pop', 'prop_hh_only_elderly','prop_managers','prop_uni_graduates']].values
bw = mgwr.sel_bw.Sel_BW(coords, y, X).search() # almost singular matrix? Check multicollinearity
gwr_model = GWR(coords, y, X, bw)
gwr_results = gwr_model.fit()
params = gwr_results.params  # This gives the estimated coefficients for each variables
gwr_results.summary()

X_df = pd.DataFrame(X, columns=['prop_young_pop', 'prop_foreign_pop', 'prop_hh_only_elderly','prop_managers',
           'prop_low_wage','prop_uni_graduates' ])  
# Compute VIF for each variable
vif_data = pd.DataFrame()
vif_data["Feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
print(vif_data)



# LASSO 
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X, y)
selected_features = X.columns[lasso.coef_ != 0]

# ADD LANDFILL/ARAKAWA DUMMIES #
print(rdata[['landfill', 'arakawa']].corr())
print(rdata['landfill'].value_counts())
print(rdata['arakawa'].value_counts())

variables = ['prop_young_pop', 'prop_o65_pop', 'prop_foreign_pop', 'prop_hh_only_elderly',
             'prop_managers', 'prop_low_wage', 'prop_uni_graduates', 'log_price', 'landfill', 'arakawa']
X_temp = rdata[variables].dropna().values  # Ensure no NaNs
vif = [variance_inflation_factor(X_temp, i) for i in range(X_temp.shape[1])]

print(pd.DataFrame({'Variable': variables, 'VIF': vif}))

print(rdata[['landfill', 'arakawa']].corr())
print(rdata.groupby(['landfill', 'arakawa']).size())

gdf = gpd.GeoDataFrame(rdata, geometry='geometry')
gdf.plot(column='landfill', cmap='coolwarm', legend=True)
plt.title("Spatial Distribution of 'landfill'")
plt.show()

gdf.plot(column='arakawa', cmap='coolwarm', legend=True)
plt.title("Spatial Distribution of 'arakawa'")
plt.show()

# Multilevel GWR -> each variable has its own bandwith ############################################
