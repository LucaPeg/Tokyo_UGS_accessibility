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
from libpysal.weights import KNN
from libpysal import weights
from spreg import OLS, ML_Lag
import esda
from libpysal.weights import KNN
from splot.esda import moran_scatterplot
from mgwr.sel_bw import Sel_BW
from scipy.spatial import distance_matrix
from mgwr.gwr import GWR, MGWR
from mgwr.utils import truncate_colormap
import libpysal
from libpysal.weights import Queen
from spreg import ML_Lag
from sklearn.model_selection import KFold
import mgwr # to import this I changed the code of spatial/Lib/site-packages/libpysal/cg/kdtree.py
# change import statement to "from scipy import inf" to "from numpy import inf"
from matplotlib.colors import LinearSegmentedColormap
from mgwr.gwr import GWR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from accessibility_functions import plot_census_points_with_basemap
np.float = float  # temporary fix for deprecated np.float 

# defining functions
def plot_local_fit(model_results):
    """Given model results it plots the local fit (R2)"""
    results.loc[:,'locR2'] =  model_results.localR2
    print(f"There are {results[results['locR2']<0]['KEY_CODE_3'].nunique()} observations with negative fit") # there are 84 rows where the model is worse than a null model
    results['locR2'] = results['locR2'].clip(lower=0) # negative values become zeroes.
    # local fit plot
    results[results['locR2']>=0].plot('locR2', legend = True) # why do I need the >=0 ?
    ax = plt.gca()
    ax.set_title("Local fit (R2)")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()

def plot_bandwidth_effect(data, varlist, variable, bws=False):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.flatten()

    if bws is False:
        bws = [60, 120, 240, 400, 600, 800]

    vmins = []
    vmaxs = []
    params = data[['geometry']].copy()
    X_values = data[varlist].values
    X_w_names = pd.DataFrame(X_values, columns=varlist)
    var_index = X_w_names.columns.get_loc(variable) + 1

    # Compute GWR for each bandwidth and collect parameter values
    for bw in bws:
        gwr_model = GWR(coords, y, X_values, bw)
        gwr_results = gwr_model.fit()
        params[f'{variable}_{bw}'] = gwr_results.params[:, var_index] 
        vmins.append(params[f'{variable}_{bw}'].min())
        vmaxs.append(params[f'{variable}_{bw}'].max())

    vmin, vmax = min(vmins), max(vmaxs)  # Get global color scale limits

    # Plot each GWR result as a choropleth map
    for i, col in enumerate(params.columns[1:]):  # Skip geometry column
        params.plot(column=col, ax=ax[i], cmap='coolwarm', legend=False,
                    vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.1)
        ax[i].set_title(f"Bandwidth: {bws[i]}")
        ax[i].axis("off")

    # Add a single color bar to the right of the plots
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []  # Empty array for the ScalarMappable
    cbar = fig.colorbar(sm, ax=ax.tolist(), fraction=0.025, pad=0.04)
    cbar.set_label(f'{variable} coefficient')

    fig.suptitle(f"Effect of {variable} across different bandwidths", fontsize=16, y=1.05)
    fig.subplots_adjust(right=0.9)  # More space for the color bar on the right
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout for better spacing

    plt.show()

def check_bandwidths(y, X, bandwidths):
    results = []
    for bw in bandwidths:
        model = GWR(coords, y, X, bw)
        results_bw = model.fit()
        summary = {
            'Bandwidth': bw,
            'R2': results_bw.R2,
            'Adj_R2': results_bw.adj_R2,
            'AICc': results_bw.aicc,
            'Effective Parameters': results_bw.tr_S
        }
        results.append(summary)

    # Convert results to a DataFrame for analysis
    results_df = pd.DataFrame(results)
    return results_df

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a colormap to a specified range.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# import data
data = os.path.join("..\\data\\final\\ugs_analysis_data.gpkg")
tokyo_wards = gpd.read_file(data, layer='wards')
census = gpd.read_file(data, layer='census_for_analysis')

## DATA PROCESSING ##############################################################################
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

# SELECTING / CREATING VARIABLES FOR REGRESSION ANALYSIS ###

census['log_price'] = np.log(census['price_mean'])
census['price_std'] = (census['log_price'] - census['log_price'].mean()) / census['log_price'].std()

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
       'prop_uni_graduates', 'price_std', 'prop_15_64', 'prop_1hh', 'prop_hh_head20',
        'log_vl_ugs_accessibility_norm', 'log_full_ugs_accessibility_norm', 'KEY_CODE_3', 'geometry',
        ]


rdata = census[relevant_attributes]
rdata.loc[:,"KEY_CODE_3"] = rdata["KEY_CODE_3"].apply(str)
rdata.isna().sum()
rdata = rdata.replace(np.nan, 0) # replace NaN with zeroes 

# select numeric columns for PCA 
numeric_cols = rdata.select_dtypes(include=['number']).columns
filtered_data = rdata[numeric_cols]

# analyze regressors' distributions
plt.figure(figsize=(15, 10))
for i, column in enumerate(filtered_data.columns, 1):
    rows = int(np.ceil(len(filtered_data.columns) / 4))  
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
    rows = int(np.ceil(len(filtered_data.columns) / 4))  
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
           'prop_low_wage','prop_uni_graduates','price_std']].values
y = rdata['log_full_ugs_accessibility_norm'].values.reshape(-1, 1) # necessary for gwr (othersize size is [7126,])
bw = mgwr.sel_bw.Sel_BW(coords, y, X).search() # optimal bandwith selection
# model fitting
gwr_model = GWR(coords, y, X, bw)
gwr_results = gwr_model.fit()
gwr_results.summary()

#       DIAGNOSTICS ##########################################################################

#       Geographical variation in model split
#       Global model fit
print(f"Global basic model AIC: {gwr_results.aic}")
print(f"Global basic model AICC: {gwr_results.aicc}")
print(f"Global basic model AIC: {gwr_results.adj_R2}")

#       Local model fit
#       initialize a dataframe where I store the results of all models
results = rdata[['KEY_CODE_3', 'geometry','longitude','latitude']]
results.loc[:,'basic_locR2'] =  gwr_results.localR2 # TSS is 0 for some observations, possibly because y values are constant
plot_local_fit(gwr_results)


#       CHECK ALL VARIABLES COEFFICIENTS
filter_tc = gwr_results.filter_tvals()  # Compute t-values once
for i, var in enumerate(['intercept','prop_young_pop', 'prop_o65_pop', 'prop_foreign_pop', 
                         'prop_hh_only_elderly', 'prop_managers', 'prop_low_wage', 
                         'prop_uni_graduates', 'price_std']):
    results[f'{var}_param'] = gwr_results.params[:, i]  # Raw coefficients
    results[f'{var}_param_tc'] = filter_tc[:, i]  # Corrected t-values

#       Create subplots (each variable has 2 plots: raw & corrected)
fig, axes = plt.subplots(9, 2, figsize=(16, 32))  # 8 variables, 2 columns

for i, var in enumerate(['intercept','prop_young_pop', 'prop_o65_pop', 'prop_foreign_pop', 
                         'prop_hh_only_elderly', 'prop_managers', 'prop_low_wage', 
                         'prop_uni_graduates', 'price_std']):
#       Row index for each variable
    row = i  

#       Plot raw GWR coefficients
    results.plot(f'{var}_param', ax=axes[row, 0], legend=True,
                 edgecolor='black', alpha=0.65, linewidth=0.5)
    axes[row, 0].set_title(f'Raw GWR Coefficients: {var}')
    axes[row, 0].axis('off')

#       Plot corrected coefficients (non-significant areas in grey)
    results.plot(f'{var}_param', ax=axes[row, 1], legend=True,
                 edgecolor='black', alpha=0.65, linewidth=0.5)
    results[results[f'{var}_param_tc'] == 0].plot(color='grey', ax=axes[row, 1], 
                                                  edgecolor='black', linewidth=0.5)  
    axes[row, 1].set_title(f'Corrected GWR Coefficients: {var}')
    axes[row, 1].axis('off')

# Global title
#plt.suptitle('GWR Results: Raw vs. Corrected Coefficients', fontsize=18)
plt.tight_layout()
plt.show()

# Global Multicollinearity check
X_df = pd.DataFrame(X, columns=['prop_young_pop', 'prop_o65_pop', 'prop_foreign_pop', 'prop_hh_only_elderly','prop_managers',
           'prop_low_wage','prop_uni_graduates','price_std', ])  
# regular VIF -> Probably this does not matter? 
vif_data = pd.DataFrame()
vif_data["Feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
print(vif_data)

# Local multicollinearity check
LCC, VIF, CN, VDP = gwr_results.local_collinearity() # this takes a long time (76 minutes)
# the following is taken from the mgwr paper
names = ['Foreign Born vs. African American', # change these
          'Foreign Born vs. Rural',
          'African  American vs. Rural']
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
for col in range(3):
   results['basic_vif'] = LCC[:, col]
   results.plot('basic_vif', ax = ax[col], legend = True)
   ax[col].set_title('LCC: ' + names[col])
   ax[col].get_xaxis().set_visible(False)
   ax[col].get_yaxis().set_visible(False)

names = ['Foreign Born vs. African American',
          'Foreign Born vs. Rural',
          'African  American vs. Rural']
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
for col in range(3):
   results['basic_vif'] = VIF[:, col]
   results.plot('vif', ax = ax[col], legend = True)
   ax[col].set_title('VIF: ' + names[col])
   ax[col].get_xaxis().set_visible(False)
   ax[col].get_yaxis().set_visible(False)
   
fig, ax = plt.subplots(1, 1, figsize = (4, 4))
results['basic_cn'] = CN
results.plot('basic_cn', legend = True, ax = ax)
ax.set_title('Condition Number')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
names = ['Foreign Born vs. African American',
          'Foreign Born vs. Rural',
          'African  American vs. Rural']
fig, ax = plt.subplots(1, 4, figsize = (16, 4))
for col in range(4):
   results['basic_vdp'] = VDP[:, col]
   results.plot('basic_vdp', ax = ax[col], legend = True)
   ax[col].set_title('VDP: ' + names[col])
   ax[col].get_xaxis().set_visible(False)
   ax[col].get_yaxis().set_visible(False)

# REDUCING MULTICOLLINEARITY (possibly overfitting) ###############################################################
# To reduce multicollinearity I implement two solutions:
# 1. Subset selection (how though? Ideally Lasso or Stepwise, but should I test it in the global regression?)
# 2. PCA 

# PCA #############################################################################################
Xpca = rdata[['PC1', 'PC2', 'PC3']].values
bw_pca = mgwr.sel_bw.Sel_BW(coords, y, Xpca).search()
gwr_pca_model = GWR(coords, y, Xpca, bw_pca)
gwr_pca_results = gwr_pca_model.fit()
gwr_pca_results.summary()

# PCA evaluate local fit
results.loc[:,'pca_locR2'] =  gwr_pca_results.localR2 # TSS is 0 for some observations, possibly because y values are constant
plot_local_fit(gwr_pca_results)

# PCA check variable coefficients
pca_filter_tc = gwr_pca_results.filter_tvals()  # Compute t-values once
for i, var in enumerate(['PC1','PC2','PC3']):
    results[f'{var}_param'] = gwr_pca_results.params[:, i]  # Raw coefficients
    results[f'{var}_param_tc'] = pca_filter_tc[:, i]  # Corrected t-values

# Create subplots (each variable has 2 plots: raw & corrected)
fig, axes = plt.subplots(3, 2, figsize=(16, 32))  # 8 variables, 2 columns

for i, var in enumerate(['PC1','PC2','PC3']):
    # Row index for each variable
    row = i  

    # Plot raw GWR coefficients
    results.plot(f'{var}_param', ax=axes[row, 0], legend=True,
                 edgecolor='black', alpha=0.65, linewidth=0.5)
    axes[row, 0].set_title(f'Raw GWR Coefficients: {var}')
    axes[row, 0].axis('off')

    # Plot corrected coefficients (non-significant areas in grey)
    results.plot(f'{var}_param', ax=axes[row, 1], legend=True,
                 edgecolor='black', alpha=0.65, linewidth=0.5)
    results[results[f'{var}_param_tc'] == 0].plot(color='grey', ax=axes[row, 1], 
                                                  edgecolor='black', linewidth=0.5)  
    axes[row, 1].set_title(f'Corrected GWR Coefficients: {var}')
    axes[row, 1].axis('off')

# Global title
#plt.suptitle('GWR Results: Raw vs. Corrected Coefficients', fontsize=18)
plt.tight_layout()
plt.show()

# CHECK VIF with PCA
X_pca_df = pd.DataFrame(Xpca, columns=['PC1','PC2','PC3'])
vif_pca =  pd.DataFrame()
vif_pca["Feature"] = X_pca_df.columns
vif_pca["VIF"] = [variance_inflation_factor(X_pca_df.values, i) for i in range(X_pca_df.shape[1])]
print(vif_pca)

# Reduced model (RED) #################################################################################
X_red = rdata[['prop_foreign_pop', 'prop_young_pop' ,'prop_hh_only_elderly',
               'prop_managers','price_std']].values
#bw_red = mgwr.sel_bw.Sel_BW(coords, y, X_red).search() 
gwr_red_model = GWR(coords, y, X_red, 4000) # I substituted bw_red with 200
gwr_red_results = gwr_red_model.fit()
params_red = gwr_red_results.params  # This gives the estimated coefficients for each variables
gwr_red_results.summary()

X_red_df = pd.DataFrame(X_red, columns=['prop_foreign_pop', 'prop_young_pop' ,'prop_hh_only_elderly',
               'prop_managers','price_std' ])

# RED Compute VIF for each variable
vif_red_data = pd.DataFrame()
vif_red_data["Feature"] = X_red_df.columns
vif_red_data["VIF"] = [variance_inflation_factor(X_red_df.values, i) for i in range(X_red_df.shape[1])]
print(vif_red_data)

# RED evaluate local fit
results.loc[:,'red_locR2'] =  gwr_red_results.localR2 # TSS is 0 for some observations, possibly because y values are constant
plot_local_fit(gwr_red_results)

# local coefficients
red_filter_tc = gwr_red_results.filter_tvals()  # Compute t-values once
for i, var in enumerate(['intercept','prop_foreign_pop','prop_young_pop','prop_hh_only_elderly', 'prop_managers','price_std']):
    results[f'{var}_param'] = gwr_red_results.params[:, i]  # Raw coefficients
    results[f'{var}_param_tc'] = red_filter_tc[:, i]  # Corrected t-values

# Create subplots (each variable has 2 plots: raw & corrected)
fig, axes = plt.subplots(6, 2, figsize=(16, 32))  # 8 variables, 2 columns

for i, var in enumerate(['intercept','prop_foreign_pop','prop_young_pop','prop_hh_only_elderly', 'prop_managers','price_std']):
    # Row index for each variable
    row = i  

    # Plot raw GWR coefficients
    results.plot(f'{var}_param', ax=axes[row, 0], legend=True,
                 edgecolor='black', alpha=0.65, linewidth=0.5)
    axes[row, 0].set_title(f'Raw GWR Coefficients: {var}')
    axes[row, 0].axis('off')

    # Plot corrected coefficients (non-significant areas in grey)
    results.plot(f'{var}_param', ax=axes[row, 1], legend=True,
                 edgecolor='black', alpha=0.65, linewidth=0.5)
    results[results[f'{var}_param_tc'] == 0].plot(color='grey', ax=axes[row, 1], 
                                                  edgecolor='black', linewidth=0.5)  
    axes[row, 1].set_title(f'Corrected GWR Coefficients: {var}')
    axes[row, 1].axis('off')

# Global title
#plt.suptitle('GWR Results: Raw vs. Corrected Coefficients', fontsize=18)
plt.tight_layout()
plt.show()


# regularized GWR - Apply ridge regression-style penalties to smooth coefficients and prevent overfitting

##################################################################################################
# MULTISCALE GWR -> each variable has its own bandwith ###########################################

# RED REGRESSION #################################################################################
X_red_st = (X_red - X_red.mean(axis = 0)) / X_red.std(axis = 0)
y_st = (y - y.mean(axis = 0)) / y.std(axis = 0)

# Add intercept
X_red_st = np.hstack([np.ones((X.shape[0], 1)), X_red_st])  # Adds a column of ones for the intercept | I didn't do this, but still get 6 coefficients

# bandwith selection
mgwr_red_selector = Sel_BW(coords, y_st, X_red_st, multi = True) 
mgwr_bw = mgwr_red_selector.search() # this takes such a long time (134minutes) array([ 43.,  52.,  52., 832.,  44.,  44.])

# fit model
mgwr_red_model = MGWR(coords, y_st, X_red_st, mgwr_red_selector, fixed=False)
mgwr_red_results = mgwr_red_model.fit()
mgwr_red_results.summary() # I call on mgwr_results because I modified the variable name, however mgwr_results is loaded in memory


# MGWR with PCA #################################################################################
Xpca_st = Xpca - Xpca.mean(axis=0) / Xpca.std(axis=0)
Xpca_st = np.hstack([np.ones((X.shape[0],1)), Xpca_st]) # add intercept
mgwr_pca_selector = Sel_BW(coords, y_st, Xpca_st, multi = True) 
mgwr_pca_bw = mgwr_pca_selector.search() 
mgwr_pca_model = MGWR(coords, y_st, Xpca_st, mgwr_pca_selector)
mgwr_pca_results = mgwr_pca_model.fit()
mgwr_pca_results.summary()

# Check coefficients significance locally
print(mgwr_pca_bw) # print bandwidths
results['mgwr_pca_intercept'] = mgwr_pca_results.params[:,0] 
results['mgwr_pca_PC1'] = mgwr_pca_results.params[:,1]
results['mgwr_pca_PC2'] = mgwr_pca_results.params[:,2]
results['mgwr_pca_PC3'] = mgwr_pca_results.params[:,3]
# get t-values
mgwr_pca_fil_t = mgwr_pca_results.filter_tvals()

for i, var in enumerate(['intercept','PC1','PC2','PC3']):
    results[f'{var}_param'] = mgwr_pca_results.params[:, i]  # Raw coefficients
    results[f'{var}_param_tc'] = mgwr_pca_fil_t[:, i]  # Corrected t-values

# Create subplots (each variable has 2 plots: raw & corrected)
fig, axes = plt.subplots(6, 2, figsize=(16, 32))  # 8 variables, 2 columns

for i, var in enumerate(['intercept','PC1','PC2','PC3']):
    # Row index for each variable
    row = i  

    # Plot raw GWR coefficients
    results.plot(f'{var}_param', ax=axes[row, 0], legend=True,
                 edgecolor='black', alpha=0.65, linewidth=0.5)
    axes[row, 0].set_title(f'Raw GWR Coefficients: {var}')
    axes[row, 0].axis('off')

    # Plot corrected coefficients (non-significant areas in grey)
    results.plot(f'{var}_param', ax=axes[row, 1], legend=True,
                 edgecolor='black', alpha=0.65, linewidth=0.5)
    results[results[f'{var}_param_tc'] == 0].plot(color='grey', ax=axes[row, 1], 
                                                  edgecolor='black', linewidth=0.5)  
    axes[row, 1].set_title(f'Corrected GWR Coefficients: {var}')
    axes[row, 1].axis('off')

# Global title
#plt.suptitle('GWR Results: Raw vs. Corrected Coefficients', fontsize=18)
plt.tight_layout()
plt.show()



# Compare local multicollinearity between GWR and MGWR
gwr_red_lc = gwr_red_results.local_collinearity()
mgwr_red_lc = mgwr_red_results.local_collinearity()

gwr_pca_lc = gwr_pca_results.local_collinearity()
mgwr_pca_lc = mgwr_pca_results.local_collinearity()

results['gwr_red_cn'] = gwr_red_lc[2]
results['mgwr_red_cn'] = mgwr_red_lc[0] # indexes taken from the mgwr documentation
results['gwr_pca_cn'] = gwr_pca_lc[2]
results['mgwr_pca_cn'] = mgwr_pca_lc[0]

# Create the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax0 = axes[0]
ax0.set_title('GWR "reduced" Condition Number', fontsize=10)
ax1 = axes[1]
ax1.set_title('MGWR "reduced" Condition Number', fontsize=10)

# Define the colormap
cmap = plt.cm.RdYlBu
vmin = np.min([results['gwr_red_cn'].min(), results['mgwr_red_cn'].min()])
vmax = np.max([results['gwr_red_cn'].max(), results['mgwr_red_cn'].max()])

# Truncate the colormap if necessary
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

# Plot the data
results.plot('gwr_red_cn', cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax,
             edgecolor='lightgrey', alpha=0.95, linewidth=0.75)
results.plot('mgwr_red_cn', cmap=cmap, ax=ax1, vmin=vmin, vmax=vmax,
             edgecolor='lightgrey', alpha=0.95, linewidth=0.75)

# Adjust layout and add colorbar
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []  # Required for the colorbar to work
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=10)

# Hide axes
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

plt.show()


# same plot for pca gwr and mgwr
# Create the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax0 = axes[0]
ax0.set_title('GWR PCA Condition Number', fontsize=10)
ax1 = axes[1]
ax1.set_title('MGWR PCA Condition Number', fontsize=10)

# Define the colormap
cmap = plt.cm.RdYlBu
vmin = np.min([results['gwr_pca_cn'].min(), results['mgwr_pca_cn'].min()])
vmax = np.max([results['gwr_pca_cn'].max(), results['mgwr_pca_cn'].max()])

# Truncate the colormap if necessary
if (vmin < 0) & (vmax < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
elif (vmin > 0) & (vmax > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

# Plot the data
results.plot('gwr_pca_cn', cmap=sm.cmap, ax=ax0, vmin=vmin, vmax=vmax,
             edgecolor='lightgrey', alpha=0.95, linewidth=0.75)
results.plot('mgwr_pca_cn', cmap=cmap, ax=ax1, vmin=vmin, vmax=vmax,
             edgecolor='lightgrey', alpha=0.95, linewidth=0.75)

# Adjust layout and add colorbar
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.14, 0.03, 0.75])
sm._A = []  # Required for the colorbar to work
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.tick_params(labelsize=10)

# Hide axes
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

plt.show()

# DEAL WITH OVERFITTING #############################################################
# issue with my models: they tend to overfit too much

bandwidths = np.arange(0, 5000, 10)
bw_red = check_bandwidths(y, X_red, bandwidths)

# bandwidth and adj_R2
plt.figure(figsize=(10,8))
plt.plot(bw_red['Bandwidth'], bw_red['Adj_R2'])
plt.xlabel('Bandwidth')
plt.ylabel('Adj R2')
plt.legend()
plt.grid(True)
plt.show()

# bandwidth and number of effective parameters 
plt.figure(figsize=(10,8))
plt.plot(bw_red['Bandwidth'], bw_red['Effective Parameters'])
plt.xlabel('Bandwidth')
plt.ylabel('Effective_Parameters')
plt.legend()
plt.grid(True)
plt.show()

# bandwidth and the AICc
plt.figure(figsize=(10,8))
plt.plot(bw_red['Bandwidth'], bw_red['AICc'])
plt.xlabel('Bandwidth')
plt.ylabel('AICc')
plt.legend()
plt.grid(True)
plt.show()

# EVALUATE COEFFICIENTS BY BANDWIDTH
cols_reduced = ['prop_foreign_pop', 'prop_young_pop' ,'prop_hh_only_elderly',
               'prop_managers','price_std' ]
cols_pca = ['PC1','PC2','PC3' ]

plot_bandwidth_effect(rdata, cols_reduced, 'prop_mangers', bws=[60, 200, 500, 1000, 2000, 3000])
plot_bandwidth_effect(rdata, cols_pca, 'PC1')



# DIVIDE IN TRAIN AND TEST #########################################
def compute_gwr_cv_mse(coords, y, X, bandwidths, kernel='gaussian', k=5):
    """
    Compute cross-validation MSE for different bandwidths in GWR.

    Args:
        coords (np.array): Spatial coordinates (n,2).
        y (np.array): Target variable (n,).
        X (np.array): Feature matrix (n,p).
        bandwidths (list): List of bandwidths to test.
        kernel (str): Kernel type ('gaussian' or 'bisquare').
        k (int): Number of folds for cross-validation.

    Returns:
        list: List of average MSEs for each bandwidth.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_values = []
    coords = np.array(coords)
    
    for bw in bandwidths:
        mse_list = []

        for train_idx, test_idx in kf.split(coords):
            
            coords_train, coords_test = coords[train_idx], coords[test_idx]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            gwr_model = GWR(coords_train, y_train, X_train, bw, kernel=kernel)
            gwr_results = gwr_model.fit()

                        # Make predictions using the trained GWR model
            pred_results = gwr_model.predict(coords_test, X_test)
            y_pred = pred_results.predictions.flatten()  # Extract predicted values

            # Compute MSE
            mse = np.mean((y_test - y_pred) ** 2)
            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        mse_values.append(avg_mse)
        print(f"Bandwidth: {bw}, MSE: {avg_mse:.5f}")

    return mse_values

# Define bandwidths to test
bandwidths1 = np.arange(50, 2500, 100)  # Test bandwidths from 500 to 5000 in steps of 500

# Compute MSE for each bandwidth
mse_values = compute_gwr_cv_mse(coords, y, X_red, bandwidths1)

# Plot MSE vs. Bandwidth
plt.figure(figsize=(8,5))
plt.plot(bandwidths1, mse_values, marker='o', linestyle='-')
plt.xlabel("Bandwidth")
plt.ylabel("Cross-Validation MSE")
plt.title("GWR Bandwidth Selection via CV")
plt.grid(True)
plt.show()

# TODO | THINGS I WANT TO TRY DOING
# set minimum bandwidth in automatic selection | Try with non-adaptive too!

# Mixed measure: both mse and prop of significant coefficients

def compute_gwr_cv_metrics(coords, y, X, bandwidths, kernel='gaussian', k=5, exclude_intercept=True):
    """
    For each candidate bandwidth, perform k-fold CV and compute:
      - Average MSE on the test folds
      - Average proportion of statistically significant coefficients
        (based on the filtered t-values from each local fit)
    
    Args:
        coords (list or np.array): Spatial coordinates (n,2) as list of tuples or array.
        y (np.array): Target variable (n,).
        X (np.array): Feature matrix (n,p).
        bandwidths (iterable): Candidate bandwidth values.
        kernel (str): 'gaussian' or 'bisquare'.
        k (int): Number of folds.
        exclude_intercept (bool): If True, compute significance only on the explanatory variables (exclude intercept).
    
    Returns:
        mse_values (list): Average MSE for each bandwidth.
        signif_values (list): Average proportion of significant coefficients for each bandwidth.
    """
    # Ensure coords is an np.array
    coords_arr = np.array(coords)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_values = []
    signif_values = []
    
    for bw in bandwidths:
        fold_mse = []
        fold_signif = []
        
        for train_idx, test_idx in kf.split(coords_arr):
            # Extract train/test data using the indices from splitting coords
            coords_train = coords_arr[train_idx]
            coords_test = coords_arr[test_idx]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit the model on the training set
            gwr_model = GWR(coords_train, y_train, X_train, bw, kernel=kernel)
            gwr_results = gwr_model.fit()
            
            # Use the model's predict method for out-of-sample predictions.
            # Note: predict() is a method on the GWR model instance.
            pred_results = gwr_model.predict(coords_test, X_test)
            y_pred = pred_results.predictions.flatten()
            
            # Compute the MSE for this fold
            mse_fold = np.mean((y_test.flatten() - y_pred)**2)
            fold_mse.append(mse_fold)
            
            # Compute the proportion of significant coefficients.
            # Use filter_tvals() to set non-significant t-values to 0.
            # gwr_results.filter_tvals() returns an array of "corrected" t-values.
            tvals_corr = gwr_results.filter_tvals()
            # Option: Exclude the intercept (first column) if desired:
            if exclude_intercept:
                tvals_corr = tvals_corr[:, 1:]
            
            # Determine significance: here we assume that a t-value of 0 means non-significant.
            # (This is how filter_tvals() is designed.)
            signif_mask = tvals_corr != 0
            prop_signif = np.mean(signif_mask)  # overall proportion of significant coefficients in this fold
            fold_signif.append(prop_signif)
        
        # Average over folds for this bandwidth:
        mse_values.append(np.mean(fold_mse))
        signif_values.append(np.mean(fold_signif))
        print(f"Bandwidth: {bw}, CV MSE: {np.mean(fold_mse):.5f}, % Significant: {np.mean(fold_signif)*100:.1f}%")
    
    return mse_values, signif_values

bandwidths2 = np.arange(50, 5000, 500)

# Compute CV metrics
mse_vals, signif_vals = compute_gwr_cv_metrics(coords, y, X_red, bandwidths2, kernel='gaussian', k=5)

# Plotting: We'll create a dual-axis plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Bandwidth')
ax1.set_ylabel('CV MSE', color=color)
ax1.plot(bandwidths2, mse_vals, marker='o', color=color, label='CV MSE')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # create a second y-axis that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('% Significant Coefficients', color=color)
# Convert proportion to percentage:
ax2.plot(bandwidths, np.array(signif_vals)*100, marker='s', color=color, label='% Significant')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('CV MSE and Local Significance vs. Bandwidth')
plt.show()