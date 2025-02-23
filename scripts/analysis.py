# Import libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import fiona
import math
import seaborn as sns
import matplotlib as mpl
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
from sklearn.utils import resample
import esda
from libpysal.weights import KNN
from libpysal.weights import W
import statsmodels.api as sm
from esda.moran import Moran
from scipy.stats import shapiro
np.float = float  # temporary fix for deprecated np.float 

# defining functions

def run_gwr(data, varlist, bw, y='full'):
    coords = list(zip(data['longitude'], data['latitude']))
    X_values = np.hstack([np.ones((len(data), 1)), data[varlist].values])  # Add intercept
    if y == 'full':
        y = data['log_full_ugs_accessibility_norm'].values.reshape(-1, 1)
    elif y == 'vl':
        y = data['log_vl_ugs_accessibility_norm'].values.reshape(-1, 1)
    else: 
        raise ValueError("y must be 'full")
    gwr_results = GWR(coords, y, X_values, bw).fit()
    return gwr_results

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
        bws = [60, 120, 240, 600, 1000, 2000]

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
        params.plot(column=col, ax=ax[i], cmap='viridis', legend=False,
                    vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.1)
        ax[i].set_title(f"Bandwidth: {bws[i]}")
        ax[i].axis("off")

    # Add a single color bar to the right of the plots
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
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

def plot_local_coeff(data, varlist, bw):
    """Takes a gdf, list of variables, and bandwidth. Plots the local coefficients
    (excluding the intercept from plots).
    """
    coords = list(zip(data['longitude'], data['latitude']))
    X_values = np.hstack([np.ones((len(data), 1)), data[varlist].values])  # Add intercept
    y = data['log_full_ugs_accessibility_norm'].values.reshape(-1, 1)
    gwr_results = GWR(coords, y, X_values, bw).fit()

    results_df = data[['KEY_CODE_3', 'geometry', 'longitude', 'latitude']]
    filter_tc = gwr_results.filter_tvals()  # Compute t-values once

    # Loop only over variables (excluding the intercept)
    for i, var in enumerate(varlist):
        results_df[f'{var}_param'] = gwr_results.params[:, i + 1]  # Skip intercept
        results_df[f'{var}_param_tc'] = filter_tc[:, i + 1]  # Skip intercept

    # Create subplots (each variable has 2 plots: raw & corrected)
    fig, axes = plt.subplots(len(varlist), 2, figsize=(16, 4 * len(varlist)))

    for i, var in enumerate(varlist):
        # Plot raw GWR coefficients
        results_df.plot(f'{var}_param', ax=axes[i, 0], legend=True,
                        edgecolor='black', alpha=0.65, linewidth=0.5)
        axes[i, 0].set_title(f'Raw GWR Coefficients: {var}')
        axes[i, 0].axis('off')

        # Plot corrected coefficients (non-significant areas in grey)
        results_df.plot(f'{var}_param', ax=axes[i, 1], legend=True,
                        edgecolor='black', alpha=0.65, linewidth=0.5)
        results_df[results_df[f'{var}_param_tc'] == 0].plot(color='grey', ax=axes[i, 1], 
                                                            edgecolor='black', linewidth=0.5)  
        axes[i, 1].set_title(f'Corrected GWR Coefficients: {var}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def compute_multisignificance(data, varlist, bws, y='full'):
    """
    Computes adjusted R2 and the percentage of statistically significant coefficients
    for each variable across different bandwidths.

    Args:
        data (pd.DataFrame): Data containing coordinates, dependent, and independent variables.
        varlist (list): List of independent variable names.
        bws (list): List of bandwidths to test.
        y (str): Whether to use 'full' or 'vl' as the dependent variable.

    Returns:
        pd.DataFrame: A dataframe with adjusted R2 and significance percentages for each variable.
    """
    coords = list(zip(data['longitude'], data['latitude']))  # Fixed typo
    X_values = data[varlist].values

    if y == 'full':
        y_values = data['log_full_ugs_accessibility_norm'].values.reshape(-1, 1)
    elif y == 'vl':
        y_values = data['log_vl_ugs_accessibility_norm'].values.reshape(-1, 1)
    else:
        raise ValueError("y must be 'full' or 'vl'")

    results = []

    for bw in bws:
        gwr_results = GWR(coords, y_values, X_values, bw).fit()
        R2 = gwr_results.adj_R2

        # compute % of statistically significant coefficents
        tvals_corr = gwr_results.filter_tvals()[:, 1:]  # Exclude intercept
        signif_mask = tvals_corr != 0
        signif_perc = np.mean(signif_mask, axis=0)  # Compute proportion for each variable
        results.append([bw, R2] + list(signif_perc))

        print(f"Bandwidth: {bw}, AdjR2: {R2:.5f}")

    # Convert results to DataFrame
    signif_df = pd.DataFrame(results, columns=['Bandwidth', 'AdjR2'] + varlist)
    
    return signif_df

def plot_multisignificance(signif_df):
    """
    Plots Adjusted R² and percentage of statistically significant coefficients 
    for each variable across different bandwidths.

    Args:
        signif_df (pd.DataFrame): DataFrame obtained with "compute_multisignificance()"
    Returns:
        a beautiful graph
    """
    varlist = signif_df.columns[2:]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot Adjusted R² on primary y-axis
    line1, = ax1.plot(signif_df['Bandwidth'], signif_df['AdjR2'], 'tab:blue', label='AdjR²')
    ax1.set_xlabel('Bandwidth')
    ax1.set_ylabel('Adjusted R²', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Secondary y-axis for percentage of significant coefficients
    ax2 = ax1.twinx()
    colors = plt.cm.viridis(np.linspace(0, 1, len(varlist)))  # Generate distinct colors
    lines = [line1]  # Store all lines for a single legend
    labels = ['AdjR²']

    for var, color in zip(varlist, colors):
        line, = ax2.plot(signif_df['Bandwidth'], signif_df[var] * 100, linestyle='--', color=color, label=var)  # Convert to %
        lines.append(line)
        labels.append(var)

    ax2.set_ylabel('Significant Coefficient %', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Merge legends into one
    ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.1, 0.5), borderaxespad=0)

    plt.title('Adjusted R² and % of Statistically Significant Coefficients')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def lorenz_curve(data):
    """Compute the Lorenz curve."""
    data_sorted = np.sort(data)
    cumulative = np.cumsum(data_sorted) / data_sorted.sum()
    cumulative = np.insert(cumulative, 0, 0)  # Add (0,0) to the curve
    return cumulative

def gini_coefficient(data):
    """Compute the Gini coefficient."""
    data_sorted = np.sort(data)
    n = len(data)
    cumulative = np.cumsum(data_sorted) / data_sorted.sum()
    gini_index = 1 - 2 * np.trapz(cumulative, dx=1/n)
    return gini_index

def compute_gwr_cv_metrics(coords, y, X, bandwidths, kernel='gaussian', k=5 , exclude_intercept=True):
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

def plot_mse_signif(mse, significance, bw):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Bandwidth')
    ax1.set_ylabel('CV MSE', color=color)
    ax1.plot(bandwidths, mse, marker='o', color=color, label='CV MSE')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # create a second y-axis that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('% Significant Coefficients', color=color)
    # Convert proportion to percentage:
    ax2.plot(bandwidths, np.array(significance)*100, marker='s', color=color, label='% Significant')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('CV MSE and Local Significance vs. Bandwidth')
    plt.show()

def plot_metrics_by_band(results_df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    # Adj R2 RED
    axes[0].plot(results_df['Bandwidth'], results_df['Adj_R2'], label="Adj R2", color='b')
    axes[0].set_ylabel("Adj R2")
    axes[0].legend()
    axes[0].grid(True)

    # ENP RED
    axes[1].plot(results_df['Bandwidth'], results_df['Effective Parameters'], label="Effective Parameters", color='g')
    axes[1].set_ylabel("Effective Parameters")
    axes[1].legend()
    axes[1].grid(True)

    # AICc RED
    axes[2].plot(results_df['Bandwidth'], results_df['AICc'], label="AICc", color='r')
    axes[2].set_xlabel("Bandwidth")
    axes[2].set_ylabel("AICc")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

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
census['price_std'] = (census['log_price'] - census['log_price'].min()) /(census['log_price'].max() -census['log_price'].min())

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

# compute Gini index for each accessibility measure
for measure in accessibility_measures:
    data = census[measure + '_norm'].dropna().values
    gini = gini_coefficient(data)
    print(f'Gini Coefficient for {measure}: {gini:.2f}')
    

#####################################################################################
# EXPLORATORY DATA ANALYSIS #########################################################


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
    problematic_census.update(census[census[attribute]>=1]["KEY_CODE_3"])

census = census[~census['KEY_CODE_3'].isin(problematic_census)]

# census with high population
plot_census_points_with_basemap(census, 'over', 4000) # do these locations make sense?
census[census['pop_tot']>4000] # They do make sense -> Japanese Tower Buldings

# focus on the indepedent variables
relevant_attributes = ['prop_young_pop', 'prop_o65_pop', 'prop_o75_pop', 'prop_foreign_pop',
       'prop_hh_only_elderly', 'prop_managers', 'prop_high_wage', 'prop_low_wage', 
       'prop_uni_graduates', 'price_std', 'prop_15_64', 'prop_1hh', 'prop_hh_head20',
        'log_vl_ugs_accessibility_norm', 'log_full_ugs_accessibility_norm', 'KEY_CODE_3', 'geometry']


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
naive_vars = ['prop_young_pop', 'prop_o65_pop', 'prop_foreign_pop', 'prop_hh_only_elderly','prop_managers',
           'prop_low_wage','prop_uni_graduates','price_std']

X = rdata[naive_vars].values
y = rdata['log_full_ugs_accessibility_norm'].values.reshape(-1, 1) # necessary for gwr (othersize size is [7126,])
bw = mgwr.sel_bw.Sel_BW(coords, y, X).search() # optimal bandwith selection
# model fitting
gwr_results = run_gwr(rdata, naive_vars, bw, y='full')
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

bw_red = mgwr.sel_bw.Sel_BW(coords, y, X_red).search(bw_min=500, bw_max=4000) # always chooses the lowest bound
gwr_red_model = GWR(coords, y, X_red, bw_red) # I substituted bw_red with 200
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

# bandwith selection
mgwr_red_selector = Sel_BW(coords, y_st, X_red_st, multi = True) 
mgwr_bw = mgwr_red_selector.search() # this takes such a long time (134minutes) array([ 43.,  52.,  52., 832.,  44.,  44.])

# fit model
mgwr_red_model = MGWR(coords, y_st, X_red_st, mgwr_red_selector, fixed=False)
mgwr_red_results = mgwr_red_model.fit()
mgwr_red_results.summary() # I call on mgwr_results because I modified the variable name, however mgwr_results is loaded in memory

#   extract coefficent estimates
results['mgwr_red_intercept'] = mgwr_red_results.params[:,0] 
results['mgwr_red_foreign'] = mgwr_red_results.params[:,1]
results['mgwr_red_young'] = mgwr_red_results.params[:,2]
results['mgwr_pca_elderly'] = mgwr_red_results.params[:,3]
results['mgwr_red_managers'] = mgwr_red_results.params[:,4]
results['mgwr_pca_price'] = mgwr_red_results.params[:,5]

# get t-values
mgwr_red_fil_t = mgwr_red_results.filter_tvals()

for i, var in enumerate(['intercept','prop_foreigners','prop_young','propr_elderly_hh','prop_managers','house_price']):
    results[f'{var}_param'] = mgwr_red_results.params[:, i]  # Raw coefficients
    results[f'{var}_param_tc'] = mgwr_red_fil_t[:, i]  # Corrected t-values

# Create subplots (each variable has 2 plots: raw & corrected)
fig, axes = plt.subplots(6, 2, figsize=(16, 32))  # 8 variables, 2 columns

for i, var in enumerate(['intercept','prop_foreigners','prop_young','propr_elderly_hh','prop_managers','house_price']):
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


# MGWR with PCA #################################################################################
Xpca_st = Xpca - Xpca.mean(axis=0) / Xpca.std(axis=0)
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
fig, axes = plt.subplots(4, 2, figsize=(16, 32))  # 8 variables, 2 columns

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

bandwidths = np.arange(0, 3500, 50)
bw_red = check_bandwidths(y, X_red, bandwidths)
bw_pca = check_bandwidths(y, Xpca, bandwidths)

   
plot_metrics_by_band(bw_pca)

# EVALUATE COEFFICIENTS BY BANDWIDTH
cols_reduced = ['prop_foreign_pop', 'prop_young_pop' ,'prop_hh_only_elderly',
               'prop_managers','price_std' ]
cols_pca = ['PC1','PC2','PC3' ]

plot_bandwidth_effect(rdata, cols_reduced, 'prop_hh_only_elderly', bws=[60, 200, 500, 1000, 2000, 3000])
plot_bandwidth_effect(rdata, cols_reduced, 'price_std') 
plot_bandwidth_effect(rdata, cols_pca, 'PC1')
plot_bandwidth_effect(rdata, cols_pca, 'PC3')


# CROSS VALIDATION #########################################

# Define bandwidths to test
bandwidths = np.arange(50, 3000, 100)  # Test bandwidths from 500 to 5000 in steps of 500

# Compute CV metrics
mse_vals, signif_vals = compute_gwr_cv_metrics(coords, y, X_red, bandwidths, kernel='gaussian', k=5)
plot_mse_signif(mse_vals, signif_vals, bandwidths)
pca_mse, pca_sign = compute_gwr_cv_metrics(coords, y, Xpca, bandwidths)
plot_mse_signif(pca_mse, pca_sign, bandwidths)

# Check local coeff significance
cols_pca = ['PC1', 'PC2', 'PC3']
cols_reduced = ['prop_foreign_pop', 'prop_young_pop', 'prop_hh_only_elderly', 'prop_managers', 'price_std']
plot_local_coeff(rdata, cols_pca, 500 )
plot_local_coeff(rdata, cols_pca, 2000 )
plot_local_coeff(rdata, cols_reduced, 500)
plot_local_coeff(rdata, cols_reduced, 2000)

# plot local fits
plot_local_fit(GWR(coords, y, Xpca, 500).fit())
plot_local_fit(GWR(coords, y, Xpca, 2000).fit())
plot_local_fit(GWR(coords, y, Xpca, 4000).fit())
plot_local_fit(GWR(coords, y, X_red, 500).fit())
plot_local_fit(GWR(coords, y, X_red, 2000).fit())
plot_local_fit(GWR(coords, y, X_red, 4000).fit())

# TODO after the regression check for spatial non-stationarity
# robust GWR -> for outliers
# mixed GWR -> for variables that don't vary locally
    
pca_multi = compute_multisignificance(rdata,['PC1','PC2','PC3'],bandwidths)
red_multi = compute_multisignificance(rdata, cols_reduced, bandwidths)
plot_multisignificance(pca_multi)
plot_multisignificance(red_multi)

# CHECK COEFFICIENTS' STABILITY 

def compute_coefficient_variation(data, varlist, bws, y='full'):
    coords = list(zip(data['longitude'], data['latitude']))
    X_values = data[varlist].values
    y_values = data['log_full_ugs_accessibility_norm'].values.reshape(-1, 1) if y == 'full' else data['log_vl_ugs_accessibility_norm'].values.reshape(-1, 1)
    coeffs = {var: [] for var in varlist}

    for bw in bws:
        gwr_results = GWR(coords, y_values, X_values, bw).fit()
        for i, var in enumerate(varlist):
            coeffs[var].append(gwr_results.params[:, i + 1].mean())  # Mean coefficient per bandwidth

    coeff_df = pd.DataFrame(coeffs, index=bws)
    coeff_df.plot(marker='o', linestyle='-')
    plt.xlabel("Bandwidth")
    plt.ylabel("Mean Coefficient Value")
    plt.title("Coefficient Stability Across Bandwidths")
    plt.show()
    
bandwidths = range(0,3001,25)
cols_reduced = ['prop_foreign_pop', 'prop_young_pop', 'prop_hh_only_elderly', 'prop_managers', 'price_std']
compute_coefficient_variation(rdata, cols_reduced, bandwidths)

def bootstrap_gwr(data, varlist, bw, n_bootstraps=100, y='full'):
    coeffs = {var: [] for var in varlist}

    for _ in range(n_bootstraps):
        # Resample the data
        sample = resample(data, replace=True)

        # Extract resampled coordinates, X, and y
        boot_coords = list(zip(sample['longitude'], sample['latitude']))
        boot_X = sample[varlist].values
        if y == 'full':
            boot_y = sample['log_full_ugs_accessibility_norm'].values.reshape(-1, 1)
        elif y == 'vl':
            boot_y = sample['log_vl_ugs_accessibility_norm'].values.reshape(-1, 1)
        else: 
            raise ValueError
        # Fit GWR on bootstrapped sample
        gwr_model = GWR(boot_coords, boot_y, boot_X, bw)
        gwr_results = gwr_model.fit()

        # Store mean coefficients from the sample
        for i, var in enumerate(varlist):
            coeffs[var].append(gwr_results.params[:, i + 1].mean())  # Exclude intercept

    # Convert results to DataFrame
    coeff_df = pd.DataFrame(coeffs)

    # Compute confidence intervals
    conf_intervals = coeff_df.quantile([0.025, 0.975])
    print(conf_intervals)

    return coeff_df

# Check how local coefficients change:
# RATIONALE : checking just the 0.025 and 0.975 intervals does not make to much sense:
#             as seen in the sensitivity analysis, some regressors have different sign across the study area
#             high income is positively associated to increased accessibility in central tokyo but negatively in the north
#             likely this is due to the type of the UGS contributing to accessibility: in the north they are not attractive

# with bootstrapping and spatial data I have issues associating values back to the original unit. 
# for this reason I add to rdata the "KEY_CODE_3" column

def bootstrap_local_coefficients(data, varlist, bw, y='full', n_bootstraps=100):
    """
    Performs bootstrapping to estimate local coefficients for each observation,
    using the unique identifier "KEY_CODE_3" to align results.
    
    Parameters:
      data (GeoDataFrame): Must contain 'KEY_CODE_3', 'longitude', 'latitude',
                           the dependent variable 'log_full_ugs_accessibility_norm',
                           and predictors.
      varlist (list): List of predictor variable names.
      bw (float): Bandwidth for the GWR model.
      n_bootstraps (int): Number of bootstrap iterations.
      
    Returns:
      boot_df (DataFrame): DataFrame with columns:
         - 'KEY_CODE_3'
         - One column per variable in varlist containing lists of bootstrapped coefficient estimates.
    """
    # Ensure the data is sorted by KEY_CODE_3 so we have a consistent order.
    data_sorted = data.sort_values('KEY_CODE_3').reset_index(drop=True)
    uid = data_sorted['KEY_CODE_3'].values
    n = len(data_sorted)
    
    # Create a dictionary to store lists of coefficients per observation for each variable.
    # We will store a list for each unique KEY_CODE_3.
    boot_coeffs = {var: {id_: [] for id_ in uid} for var in varlist}
    
    for b in range(n_bootstraps):
        # Resample the data (maintaining the KEY_CODE_3 for matching later)
        sample = resample(data_sorted, replace=True, n_samples=n, random_state=b)
        
        # Extract coordinates, predictors (X), and dependent variable (y)
        boot_coords = list(zip(sample['longitude'], sample['latitude']))
        boot_X = sample[varlist].values
        if y == 'full':
            boot_y = sample['log_full_ugs_accessibility_norm'].values.reshape(-1, 1)
        elif y=='vl':
            boot_y = sample['log_vl_ugs_accessibility_norm'].values.reshape(-1, 1)
        else:
            raise ValueError('y must be either "full" or "vl" (for very large parks)')
        # Fit GWR on the bootstrap sample
        gwr_model = GWR(boot_coords, boot_y, boot_X, bw)
        gwr_results = gwr_model.fit()
        
        # Attach the KEY_CODE_3 from the bootstrap sample
        sample_ids = sample['KEY_CODE_3'].values
        
        # For each observation in the bootstrap sample, store its estimated coefficients
        for i, uid_val in enumerate(sample_ids):
            for j, var in enumerate(varlist):
                # gwr_results.params: intercept is column 0, then predictors in order
                boot_coeffs[var][uid_val].append(gwr_results.params[i, j+1])
    
    # Convert the results to a DataFrame where each row corresponds to a unique KEY_CODE_3
    # and each variable column contains the list of bootstrap estimates.
    boot_results = pd.DataFrame({'KEY_CODE_3': uid})
    for var in varlist:
        # Compute the standard deviation for each KEY_CODE_3 from its list of estimates
        boot_results[var + '_std'] = boot_results['KEY_CODE_3'].apply(
            lambda id_: np.std(boot_coeffs[var][id_])
        )
    
    return boot_results

def merge_bootstrap_results(gdf, boot_results):
    """
    Merge the bootstrap results (e.g., standard deviations) back into the original GeoDataFrame.
    """
    # Assuming gdf is a GeoDataFrame with a unique KEY_CODE_3 column.
    merged = gdf.merge(boot_results, on='KEY_CODE_3', how='left')
    return merged

boot60 = bootstrap_local_coefficients(rdata, cols_reduced, 60, "full", 100)

data_polygons = rdata[['geometry','KEY_CODE_3']]

merge1 = merge_bootstrap_results(data_polygons,boot60)


merge1.plot('prop_high_wage_std', legend = True) 
ax = plt.gca()
ax.set_title("high_wage_stdde")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

def plot_local_std(data, varlist, bw, y='full'):
    """Bootstraps GWR model and plots the standard deviation of each coefficient.
    
    Args:
        data (GeoDataFrame): gdf including: id (KEY_CODE_3), geometry, variables in varlist.
        varlist (list): List of the regressors.
        bw (int): Bandwidth used for the GWR.
        y (str, optional): Accessibility index used: 'full' = all parks, 'vl'= only biggest parks. Defaults to 'full'.
    """
    boot = bootstrap_local_coefficients(data, varlist, bw, y, 100)
    data_polygons = data[['geometry', 'KEY_CODE_3']]
    df = merge_bootstrap_results(data_polygons, boot)
    
    # Determine which columns to plot: only those ending with '_std'
    plot_cols = [col for col in df.columns if col.endswith('_std')]
    n_vars = len(plot_cols)
    max_cols = 3
    n_cols = min(n_vars, max_cols)
    n_rows = math.ceil(n_vars / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    # Flatten axes for consistent indexing
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(plot_cols):
        ax = axes[i]
        # Plot without legend first.
        df.plot(column=col, cmap='viridis', legend=False, ax=ax)
        ax.set_title(f"Local Std: {col}")
        ax.axis("off")
        
        # Create a colorbar for this subplot.
        # Get min and max from the column values.
        vmin = df[col].min()
        vmax = df[col].max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm._A = []  # dummy array for ScalarMappable
        
        # Add the colorbar to the axis. Adjust fraction and pad as needed.
        cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()

# local standard deviations using bw = 60
plot_local_std(rdata, cols_reduced, 1000, y='full')



# check global std. dev. of each variable as bandwidth varies
def plot_mean_std_by_bw(data, varlist, bw_values, y='full', n_bootstraps=100):
    """
    For each candidate bandwidth in bw_values, run the bootstrap to compute local 
    standard deviations and then compute the mean standard deviation across all observations 
    for each variable.
    
    Parameters:
      data (GeoDataFrame): Spatial data with 'KEY_CODE_3', 'longitude', 'latitude', dependent variable, and predictors.
      varlist (list): List of predictor variable names.
      bw_values (iterable): A list of candidate bandwidth values.
      y (str): 'full' or 'vl' (dependent variable selection).
      n_bootstraps (int): Number of bootstrap iterations.
      
    Returns:
      results_df (DataFrame): A DataFrame with columns 'Bandwidth' and one column for each variable
                              in varlist containing the mean standard deviation across observations.
    """
    # List to store results for each bandwidth.
    results_list = []
    
    for bw in bw_values:
        # Run the bootstrap function for the given bandwidth
        boot_results = bootstrap_local_coefficients(data, varlist, bw, y=y, n_bootstraps=n_bootstraps)
        # boot_results: DataFrame with columns 'KEY_CODE_3' and var+'_std' for each variable.
        # Compute the mean standard deviation for each variable.
        mean_std = {}
        for var in varlist:
            colname = var + '_std'
            mean_std[colname] = boot_results[colname].mean()
        mean_std['Bandwidth'] = bw
        results_list.append(mean_std)
        print(f"Bandwidth: {bw} processed.")
    
    # Convert list of dictionaries to DataFrame
    results_df = pd.DataFrame(results_list)
    plt.figure(figsize=(10, 6))
    
    for var in varlist:
        colname = var + '_std'
        plt.plot(results_df['Bandwidth'], results_df[colname], marker='o', label=var)
    
    plt.xlabel('Bandwidth')
    plt.ylabel('Mean Standard Deviation')
    plt.title('Mean Coefficient Standard Deviation vs. Bandwidth')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

bw_values = range(50, 2051, 100)
plot_mean_std_by_bw(rdata, cols_reduced, bw_values, y='full', n_bootstraps=100)


# TODO run the following:
plot_local_std(rdata, cols_reduced, 2000, y='full')
plot_local_std(rdata, cols_reduced, 4000, y='full')
bw_values = range(50, 2051, 100)
plot_mean_std_by_bw(rdata, cols_reduced, bw_values, y='full', n_bootstraps=100)

cols_alt = ['prop_foreign_pop', 'prop_young_pop', 'prop_hh_only_elderly', 'prop_managers',]

alt_results = run_gwr(rdata, cols_alt, 500)
alt_results.summary()
plot_local_fit(alt_results)
plot_local_coeff(rdata, cols_alt, 500)
alt_results_vl = run_gwr(rdata, cols_alt, 500, 'vl')
plot_local_fit(alt_results_vl)
alt_results_vl.summary()

cols_reduced = ['prop_foreign_pop', 'prop_young_pop', 'prop_hh_only_elderly', 'prop_managers', 'price_std']

red_results = run_gwr(rdata, cols_reduced, 500)
red_results.summary()
red_results_vl = run_gwr(rdata, cols_reduced, 800, y='vl')
red_results_vl.summary()
plot_local_fit(red_results_vl)
plot_local_coeff(rdata, cols_reduced, 500)

vif_red = pd.DataFrame()
red_df = rdata[cols_reduced]
vif_red["Feature"] = red_df.columns
vif_red["VIF"] = [variance_inflation_factor(red_df.values, i) for i in range(red_df.shape[1])]
print(vif_red)

cols_final = ['prop_foreign_pop', 'prop_1hh', 'prop_high_wage','prop_young_pop','prop_hh_only_elderly']

def get_vif(data, columns):
    vif_df = pd.DataFrame()
    df = data[columns]
    vif_df["Feature"] = df.columns
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    print(vif_df)

get_vif(rdata, cols_alt)
rdata
pca_model = run_gwr(rdata, cols_pca, 500)
pca_model.summary()
plot_local_fit(pca_model)
plot_local_coeff(rdata, cols_pca, 500)


# FINAL MODELS: reduced and PCA
from esda.moran import Moran_Local

# REDUCED MODEL #####################################################################
# 1. Check residuals for spatial autocorrelation:
bw = 500
red_model = run_gwr(rdata, cols_reduced, bw)
pca_model = run_gwr(rdata, cols_pca, 500)

def get_diagnostics(data, model, bw):
    """Performs regression diagnostics:
    1. Residuals spatial autocorrelation with Moran I
    2. QQ plot of residuals and Shapiro-Wilk test
    3. Residuals heteroscedasticity: Breusch Pagan test
    
    Args:
        data (_gdf_): Geodataframe containing longitude and latitude
        model (_gwr_model_): obtained by running run_gwr
        bw (_integer_): Bandwidth used in the GWR model
    """
    # check spatial autocollinearity
    coords = list(zip(data['longitude'], data['latitude']))
    knn = KNN.from_array(coords, k=bw)
    w = W(knn.neighbors)
    residuals = model.resid_response.flatten()  # Ensure 1D array
    # Compute Moran’s I
    moran = Moran(residuals, w)
    print(f"Moran’s I: {moran.I:.3f}, p-value: {moran.p_sim:.3f}") # almost no clustering in the residuals

    # Plot significant clusters (p < 0.05)
    local_moran = Moran_Local(residuals, w)
    coords = np.array(coords)
    plt.scatter(coords[:, 0], coords[:, 1], c=local_moran.q, cmap='viridis', s=1)
    plt.title("Local Moran's I Clusters")
    plt.colorbar(label="Cluster Type")
    plt.show()
    #1: High-High (high residuals surrounded by high residuals)
    #2: Low-Low (low residuals surrounded by low residuals)
    #3: High-Low (high residuals surrounded by low residuals)
    #4: Low-High (low residuals surrounded by high residuals)

    # 2. Check for residuals normality
    # qq plot
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q Plot of Residuals') 
    plt.show()
    # shapiro wilk test
    stat, p = shapiro(residuals)
    print(f"Shapiro-Wilk p-value: {p:.3f}")  # p > 0.05 → residuals normal
    if p > 0.05:
        print("The residuals are normally distributed")
    else: 
        print("The residuals are not normally distributed")

    # 3. Residuals heterosscedasticity

    # plot residuals v. fitted
    plt.scatter(model.predy, residuals, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted')
    plt.show()

    # Breusch-Pagan test
    import statsmodels.stats.api as sms

    bp_test = sms.het_breuschpagan(residuals, model.X)
    print(f"Breusch-Pagan p-value: {bp_test[1]:.3f}")  # p < 0.05 → heteroscedasticity
    if bp_test[1] < 0.05:
        print("The residuals are heteroscedastic")
    else:
        print("The residuals are homoscedastic")
    
get_diagnostics(rdata, red_model, 500)
get_diagnostics(rdata, pca_model, 500)


rdata['price_std'].plot(kind='hist', bins=100)