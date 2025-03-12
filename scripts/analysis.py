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
from esda.moran import Moran_Local
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
    coords = list(zip(data['longitude'], data['latitude']))
    
    y = data['log_full_ugs_accessibility_norm'].values.reshape(-1, 1)
    
    vmins = []
    vmaxs = []
    params = data[['geometry']].copy()
    X_values = data[varlist].values
    X_w_names = pd.DataFrame(X_values, columns=varlist)
    var_index = X_w_names.columns.get_loc(variable) + 1  # +1 to account for the intercept
    
    # Compute GWR for each bandwidth and collect parameter values
    for bw in bws:
        gwr_model = GWR(coords, y, X_values, bw)
        gwr_results = gwr_model.fit()
        params[f'{variable}_{bw}'] = gwr_results.params[:, var_index] 
        vmins.append(params[f'{variable}_{bw}'].min())
        vmaxs.append(params[f'{variable}_{bw}'].max())
    
    vmin, vmax = min(vmins), max(vmaxs)  # Global color scale limits
    
    # Create subplots dynamically (max 3 columns)
    n_vars = len(params.columns) - 1  # Exclude geometry
    max_cols = 3
    n_cols = min(n_vars, max_cols)
    n_rows = math.ceil(n_vars / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Skip the geometry column when plotting
    plot_cols = [col for col in params.columns if col != 'geometry']
    for i, col in enumerate(plot_cols):
        ax = axes[i]
        params.plot(column=col, cmap='viridis', legend=False, ax=ax,
                    vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.1)
        ax.set_title(f"Bandwidth: {bws[i]}")
        ax.axis("off")
    
    # Hide any extra axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    # Add a single colorbar to the figure
    import matplotlib as mpl
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []  # Dummy array for the ScalarMappable
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.04)
    cbar.set_label(f'{variable} coefficient')
    
    fig.suptitle(f"Effect of {variable} across different bandwidths", fontsize=16, y=1.05)
    fig.subplots_adjust(right=0.9)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
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
plot_local_coeff(rdata, cols_pca, 500)
plot_local_coeff(rdata, cols_pca, 2000)
plot_local_coeff(rdata, cols_reduced, 500)
plot_local_coeff(rdata, cols_reduced, 2000)

# plot local fits
plot_local_fit(GWR(coords, y, Xpca, 500).fit())
plot_local_fit(GWR(coords, y, Xpca, 2000).fit())
plot_local_fit(GWR(coords, y, Xpca, 4000).fit())
plot_local_fit(GWR(coords, y, X_red, 500).fit())
plot_local_fit(GWR(coords, y, X_red, 2000).fit())
plot_local_fit(GWR(coords, y, X_red, 4000).fit())
    
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
    plt.title("Coefficient stability across bandwidths")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
bandwidths = range(0,3001,50)
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
    max_cols = 4
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
plot_local_std(rdata, cols_reduced, 88, y='full')
plot_local_std(rdata, cols_reduced, 500, y='full')
plot_local_std(rdata, cols_reduced, 1500, y='full')

plot_local_std(rdata, cols_pca, 50, y='full')
plot_local_std(rdata, cols_pca, 500, y='full')
plot_local_std(rdata, cols_pca, 1500, y='full')



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
plot_mean_std_by_bw(rdata, cols_pca, bw_values, y='full', n_bootstraps=100) 

# DIAGNOSTICS FUNCTIONS
def get_vif(data, columns):
    vif_df = pd.DataFrame()
    df = data[columns]
    vif_df["Feature"] = df.columns
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    print(vif_df)
def get_diagnostics(data, model, bw):
    """Performs regression diagnostics:
    1. Residuals spatial autocorrelation with Moran I
    2. QQ plot of residuals and Shapiro-Wilk test
    3. Residuals heteroscedasticity: Breusch Pagan test
    4. CooksD x Leverage
    Args:
        data (_gdf_): Geodataframe containing longitude and latitude
        model (_gwr_model_): obtained by running run_gwr
        bw (_integer_): Bandwidth used in the GWR model
    """
    # check spatial autocollinearity
    coords = list(zip(data['longitude'], data['latitude']))
    knn = KNN.from_array(coords, k=bw)
    w = W(knn.neighbors)
    residuals = model.resid_response.flatten() 
    moran = Moran(residuals, w) # Compute Moran’s I
    print(f"Moran’s I: {moran.I:.6f}, p-value: {moran.p_sim:.3f}") # almost no clustering in the residuals

    # Plot significant clusters (p < 0.05)
    local_moran = Moran_Local(residuals, w)
    coords = np.array(coords)
    plt.scatter(coords[:, 0], coords[:, 1], c=local_moran.q, cmap='viridis', s=1)
    plt.title("Local Moran's I Clusters")
    plt.colorbar(label="Cluster Type")
    plt.axis('off')
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
        
        
    cooks = model.cooksD
    leverage = model.influ
    # Get number of observations (n) and number of parameters (p)
    n = len(model.y)
    #p = red_model.params.shape[1]  # p includes intercept
    p = int(model.ENP)
    # Define thresholds:
    cook_threshold = 4 / (n - p - 1)
    lev_threshold = 2 * p / n  # or use 2*(p+1)/n if preferred

    # Create scatter plot of leverage vs. Cook's distance
    plt.figure(figsize=(8, 6))
    plt.scatter(leverage, cooks, alpha=0.7, edgecolor='k')
    plt.xlabel("Leverage")
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance vs. Leverage")

    # Add horizontal line for Cook's distance threshold and vertical line for leverage threshold
    plt.axhline(y=cook_threshold, color='red', linestyle='--', 
                label=f"Cook's threshold: {cook_threshold:.3f}")
    plt.axvline(x=lev_threshold, color='blue', linestyle='--', 
                label=f'Leverage threshold: {lev_threshold:.3f}')
    plt.legend()
    plt.show()


# FINAL MODELS
cols_reduced = ['prop_foreign_pop', 'prop_young_pop' ,'prop_hh_only_elderly', 'prop_managers']
cols_pca = ['PC1', 'PC2', 'PC3']

# reduced model full
red_model = run_gwr(rdata, cols_reduced, 600)
red_model.summary()
get_diagnostics(rdata, red_model, 600)
get_vif(rdata, cols_reduced)
plot_local_fit(red_model)
plot_local_coeff(rdata, cols_reduced, 350)

red_model_vl = run_gwr(rdata, cols_reduced, 600, y='vl')
red_model_vl.summary()
get_diagnostics(rdata, red_model_vl, 600)
plot_local_fit(red_model_vl)


# PCA MODEL
pca_model = run_gwr(rdata, cols_pca, 500)
pca_model.summary()
plot_local_fit(pca_model)
plot_local_coeff(rdata, cols_pca, 500)
get_diagnostics(rdata, pca_model, 500)
get_vif(rdata, cols_pca)

pca_model_vl = run_gwr(rdata, cols_pca, 500, y='vl')
pca_model_vl.summary()
get_diagnostics(rdata, pca_model_vl, 500)
plot_local_fit(pca_model_vl)

# POST- LEVERAGE EVALUATION
# REDUCED MODEL

# Extract high leverage points
red_leverage = red_model.influ  # Extract leverage value
high_leverage_indices = np.where(red_leverage > 0.3)[0] # set 0.3 as effective thresholds
high_leverage_points = rdata.iloc[high_leverage_indices]['KEY_CODE_3'].values

# Extract high cook pointrs
red_cooks = red_model.cooksD
high_cooks_indices = np.where(red_cooks > 0.03)[0] # set 0.03 as effective threshold
high_cooks_points = rdata.iloc[high_cooks_indices]['KEY_CODE_3'].values

outlier_ids = np.union1d(high_leverage_points, high_cooks_points)

# Create a new dataset excluding these outliers
rdata_noout = rdata[~rdata['KEY_CODE_3'].isin(outlier_ids)]
print("Filtered dataset shape:", rdata_noout.shape)

# Now run the GWR on the filtered dataset
red_model_noout = run_gwr(rdata_noout, cols_reduced, 600)
red_model_noout.summary()
get_diagnostics(rdata_noout, red_model_noout, 600)
plot_local_fit(red_model_noout)
red_model_noout_vl = run_gwr(rdata_noout, cols_reduced, 600, y='vl')
red_model_noout_vl.summary()
get_diagnostics(rdata_noout, red_model_noout_vl, 600)

# where are the outliers?
highest_cook = np.where(red_cooks>0.4)[0]
highest_cook_code = rdata.iloc[highest_cook]['KEY_CODE_3'].values
outliers = rdata[rdata['KEY_CODE_3'].isin(outlier_ids)]
outliers[['prop_foreign_pop','prop_young_pop','prop_hh_only_elderly','prop_managers']]
import contextily as ctx
def plot_census_points_with_basemap(
    data,
    census_list,
    buffer_factor=1.5,
    color="red",
    markersize=10,
    alpha=0.7,
    zoom=10,
    basemap_source=ctx.providers.CartoDB.Positron,
    title=None
):
    """
    Plot census points on a basemap give KEY_CODE_3 list.

    Parameters:
        data (GeoDataFrame): census gdf (must include geometries).
        census_list (list): list of KEY_CODE_3
        buffer_factor (float): Factor to expand map bounds for zooming out.
        title (str): Title for the plot.
        color (str): Color of the points.
        markersize (int): Size of the points.
        alpha (float): Transparency of the points.
        basemap_source: Contextily basemap source.
        zoom (int): Optional zoom level for the basemap.
    """
    census_list = list(census_list)
    gdf = data[data["KEY_CODE_3"].isin(census_list)]
    
    
    if gdf.crs != "EPSG:3857":
        gdf = gdf.to_crs(epsg=3857)
    
    # get bounds so I can increase them when I have small area
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    

    # Expand the bounds to zoom out
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]

    expanded_bounds = [
        bounds[0] - x_range * (buffer_factor - 1),  # Min X
        bounds[1] - y_range * (buffer_factor - 1),  # Min Y
        bounds[2] + x_range * (buffer_factor - 1),  # Max X
        bounds[3] + y_range * (buffer_factor - 1),  # Max Y
    ]

    # Plot the census points
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.plot(
        ax=ax,
        color=color,
        markersize=markersize,
        alpha=alpha,
    )

    # Add the basemap
    ctx.add_basemap(ax, source=basemap_source, zoom=zoom)

    # Set the expanded bounds as limits
    ax.set_xlim(expanded_bounds[0], expanded_bounds[2])  # Set x-axis limits
    ax.set_ylim(expanded_bounds[1], expanded_bounds[3])  # Set y-axis limits

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.axis("off")  # Turn off axis labels

    # Show the plot
    plt.show()
plot_census_points_with_basemap(rdata, outlier_ids)

# PCA MODEL 
pca_leverage = pca_model.influ  # Extract leverage value
pca_high_leverage_indices = np.where(pca_leverage > 0.25)[0] # set 0.3 as effective thresholds
pca_high_leverage_points = rdata.iloc[pca_high_leverage_indices]['KEY_CODE_3'].values
# Extract high cook pointrs
pca_cooks = pca_model.cooksD
pca_high_cooks_indices = np.where(pca_cooks > 0.013)[0] # set 0.03 as effective threshold
pca_high_cooks_points = rdata.iloc[pca_high_cooks_indices]['KEY_CODE_3'].values
pca_outlier_ids = np.union1d(pca_high_leverage_points, pca_high_cooks_points)
# Create a new dataset excluding these outliers
pca_rdata_noout = rdata[~rdata['KEY_CODE_3'].isin(pca_outlier_ids)]
# Rerun the model
pca_model_noout = run_gwr(pca_rdata_noout, cols_pca, 500)
pca_model_noout.summary()
get_diagnostics(pca_rdata_noout, pca_model_noout, 500)

pca_crazy = run_gwr(pca_rdata_noout, cols_pca, 4000)
get_diagnostics(pca_rdata_noout, pca_crazy, 4000)


rdata

# ROBUST COEFFICIENTS 

def bootstrap_gwr_coefficients(data, varlist, bw, n_bootstraps=100, y_col='full'):
    """
    Performs bootstrap resampling by unique KEY_CODE_3 (i.e. entire observation units) 
    and fits a GWR model on each replicate. Returns a dataframe with the local coefficient 
    estimates for each bootstrap replicate.
    
    Parameters:
      data        : pandas DataFrame containing your dataset.
      varlist     : list of column names to be used as independent variables.
      bw          : fixed bandwidth (or pre-selected bandwidth) for GWR.
      n_bootstraps: number of bootstrap replicates to perform.
      y_col       : column name of the dependent variable.
      "KEY_CODE_3"     : column name with the unique identifier (KEY_CODE_3).
      lon_col     : column name for the x-coordinate (longitude).
      lat_col     : column name for the y-coordinate (latitude).
    
    Returns:
      A pandas DataFrame with columns:
         - 'bootstrap' (the bootstrap replicate index)
         - KEY_CODE_3 (unique ID for each observation)
         - one column for the intercept and one for each independent variable from varlist.
    """
    
    # Get the list of unique IDs
    unique_ids = data['KEY_CODE_3'].unique()
    all_bootstrap_results = []
    
    for b in range(n_bootstraps):
        # Sample unique IDs with replacement
        sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        
        # Select all rows corresponding to the sampled unique IDs
        # (This assumes that each KEY_CODE_3 represents a complete observation)
        boot_sample = data[data["KEY_CODE_3"].isin(sampled_ids)].copy()
        # Optional: sort by the unique identifier so that the output is easier to join later
        boot_sample.sort_values(by="KEY_CODE_3", inplace=True)
        
        # Prepare the coordinates for GWR (as a list of (lon, lat) tuples)
        coords = list(zip(boot_sample['longitude'], boot_sample['latitude']))
        # Prepare the independent variables
        X = boot_sample[varlist].values
        # Prepare the dependent variable (reshaped as a column vector)
        if y_col == 'full':
            y_var = "log_full_ugs_accessibility_norm"
        elif y_col == 'vl':
            y_var = 'log_vl_ugs_accessibility_norm'
        else:
            raise ValueError("y_col can be either 'full' or 'vl'")
            
        y = boot_sample[y_var].values.reshape(-1, 1)
        
        # Fit the GWR model using the provided fixed bandwidth
        gwr_model = GWR(coords, y, X, bw)
        gwr_results = gwr_model.fit()
        
        # Create a dataframe for the coefficients for this bootstrap replicate.
        # Assume gwr_results.params is an array of shape (n_obs, n_coeff) with the first column as the intercept.
        coef_df = pd.DataFrame(gwr_results.params, 
                               columns=['Intercept'] + varlist)
        # Add the KEY_CODE_3 values (from the bootstrap sample) and the bootstrap replicate number
        coef_df["KEY_CODE_3"] = boot_sample["KEY_CODE_3"].values
        coef_df['bootstrap'] = b
        
        # Append this replicate's coefficients to the list
        all_bootstrap_results.append(coef_df)
    
    # Combine all replicates into one dataframe
    final_bootstrap_df = pd.concat(all_bootstrap_results, ignore_index=True)
    return final_bootstrap_df

red_boot_coeff =bootstrap_gwr_coefficients(rdata_noout, cols_reduced, 600, y_col='full')

def map_bootstrap_subplots(gdf_original, bootstrap_df, coef_vars, key_col='KEY_CODE_3', 
                           ncols=2, cmap='viridis'):
    """
    For each variable in coef_vars (excluding the intercept), this function:
      1. Computes per-unit median, 2.5th and 97.5th percentiles across bootstraps.
      2. Merges these summary stats back into the original GeoDataFrame.
      3. Creates a single figure with subplots for each variable showing the median coefficient.
    
    Parameters:
      gdf_original : GeoDataFrame with geometry and a unique identifier field (key_col).
      bootstrap_df : DataFrame with bootstrap coefficient results including key_col and a column "bootstrap".
      coef_vars    : list of coefficient variable names to map (e.g. ['Var1', 'Var2', ...]).
                     (Exclude 'Intercept' if not desired.)
      key_col      : unique identifier field common to both datasets.
      ncols        : number of columns for subplots (default 2).
      cmap         : colormap for the plots (default 'viridis').
    
    Returns:
      A matplotlib Figure object with the subplots.
    """
    
    # Summarize bootstrap results: group by the unique key and compute summary statistics for each variable.
    summaries = {}
    grouped = bootstrap_df.groupby(key_col)
    for var in coef_vars:
        summary = grouped[var].agg(median=lambda x: np.median(x),
                                   ci_lower=lambda x: np.percentile(x, 2.5),
                                   ci_upper=lambda x: np.percentile(x, 97.5)).reset_index()
        summaries[var] = summary

    # Determine the number of subplots needed
    nvars = len(coef_vars)
    nrows = int(np.ceil(nvars / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    # Flatten axes array for easier indexing.
    if nrows * ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # For each variable, merge the summary with the original GeoDataFrame and plot.
    for i, var in enumerate(coef_vars):
        summary = summaries[var]
        merged = gdf_original.merge(summary, on=key_col, how='left')
        
        merged.plot(column='median', ax=axes[i], cmap=cmap, legend=True,
                    legend_kwds={'label': f"Median {var} Coefficient", "shrink": 0.6})
        axes[i].set_title(f"{var}", fontsize=12)
        axes[i].set_axis_off()
    
    # Remove any extra empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    fig.tight_layout()
    plt.show()
    return fig

# Example usage:
# Assume gdf_original is your GeoDataFrame with geometry and KEY_CODE_3.
# Assume boot_coef_df is the bootstrap coefficients DataFrame from bootstrap_gwr_coefficients().
# And varlist is your list of independent variables (excluding the intercept).
fig = map_bootstrap_subplots(rdata_noout, red_boot_coeff, coef_vars=cols_reduced, key_col='KEY_CODE_3', ncols=2)



def create_summary_dict(bootstrap_df, coef_vars, key_col='KEY_CODE_3'):
    """
    Creates a summary dictionary for the bootstrap coefficients.
    
    Parameters:
      bootstrap_df : DataFrame obtained from bootstrap_gwr_coefficients().
      coef_vars    : List of coefficient variable names to summarize (e.g., ['Var1', 'Var2', ...]).
                     Exclude the intercept if not needed.
      key_col      : Unique identifier column (default 'KEY_CODE_3').
    
    Returns:
      A dictionary where keys are coefficient names and values are DataFrames with columns:
         [key_col, f"{var}_median", f"{var}_ci_lower", f"{var}_ci_upper"]
    """
    summary_dict = {}
    grouped = bootstrap_df.groupby(key_col)
    
    for var in coef_vars:
        summary = grouped[var].agg(
            median=lambda x: np.median(x),
            ci_lower=lambda x: np.percentile(x, 2.5),
            ci_upper=lambda x: np.percentile(x, 97.5)
        ).reset_index()
        # Rename columns to include the variable name as prefix.
        summary = summary.rename(columns={
            'median': f"{var}_median",
            'ci_lower': f"{var}_ci_lower",
            'ci_upper': f"{var}_ci_upper"
        })
        summary_dict[var] = summary
        
    return summary_dict

summary_dict = create_summary_dict(red_boot_coeff, coef_vars=cols_reduced, key_col='KEY_CODE_3')
def merge_and_plot_sig_coeffs(gdf_original, summary_dict, coef_vars, 
                              ncols=2, pos_color='blue', neg_color='red', nonsig_color='lightgrey',
                              legend_title=None, main_title=None):
    """
    This function merges bootstrap summary data (median, ci_lower, ci_upper) for a list of coefficient variables
    with the original GeoDataFrame and then plots subplots (one per variable) where for each unit the fill color depends on
    whether its 95% CI is entirely positive (pos_color), entirely negative (neg_color), or crosses zero (nonsig_color).
    
    Parameters:
      gdf_original : GeoDataFrame containing original geometry and the key column.
      summary_dict : Dictionary where keys are coefficient names (matching those in coef_vars) and values are DataFrames
                     with columns: [key_col, f"{var}_median", f"{var}_ci_lower", f"{var}_ci_upper"].
      coef_vars    : List of coefficient variable names to plot (exclude the intercept if desired).
      ncols        : Number of columns for subplots (default 2).
      pos_color    : Color for units with CI entirely above 0 (default 'blue').
      neg_color    : Color for units with CI entirely below 0 (default 'red').
      nonsig_color : Color for units with CI that straddles zero (default 'lightgrey').
      legend_title : Title for the legend (optional).
      main_title   : Main title for the figure (optional).
      
    Returns:
      A matplotlib Figure object with the subplots.
    """
    # Merge all summary DataFrames into a single GeoDataFrame
    merged_gdf = gdf_original.copy()
    for var in coef_vars:
        if var in summary_dict:
            merged_gdf = merged_gdf.merge(summary_dict[var], on="KEY_CODE_3", how='left')
        else:
            raise ValueError(f"No summary data provided for variable: {var}")
    
    # Determine subplot grid dimensions
    nvars = len(coef_vars)
    nrows = int(np.ceil(nvars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if nrows*ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Define colormap mapping for significance
    cmap_dict = {'positive': pos_color, 'negative': neg_color, 'nonsig': nonsig_color}
    
    for i, var in enumerate(coef_vars):
        # Classify significance based on the confidence interval for the current variable.
        lower = merged_gdf[f"{var}_ci_lower"]
        upper = merged_gdf[f"{var}_ci_upper"]
        merged_gdf['sig'] = np.select([(lower > 0) & (upper > 0),
                                       (lower < 0) & (upper < 0)],
                                      ['positive', 'negative'], default='nonsig')
        # Plot the merged GeoDataFrame; assign color based on significance classification.
        ax = axes[i]
        merged_gdf.plot(color=merged_gdf['sig'].map(cmap_dict), ax=ax)
        ax.set_title(f"{var}", fontsize=12)
        ax.set_axis_off()
    
    # Remove extra axes if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Create custom legend handles
    handles = [mpatches.Patch(color=col, label=lab) for lab, col in cmap_dict.items()]
    if legend_title is None:
        legend_title = "Significance"
    fig.legend(handles=handles, title=legend_title, loc='lower center', ncol=len(cmap_dict), frameon=True)
    
    if main_title is not None:
        fig.suptitle(main_title, fontsize=16)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    return fig

import matplotlib.patches as mpatches
fig = merge_and_plot_sig_coeffs(rdata_noout, summary_dict, coef_vars=cols_reduced, ncols=2, main_title="Significant Coefficient Maps")


# ON VERY LARGE PARKS
red_boot_coeff_vl =bootstrap_gwr_coefficients(rdata_noout, cols_reduced, 600, y_col='vl')
fig = map_bootstrap_subplots(rdata_noout, red_boot_coeff_vl, coef_vars=cols_reduced, key_col='KEY_CODE_3', ncols=2)
summary_dict_vl = create_summary_dict(red_boot_coeff_vl, coef_vars=cols_reduced, key_col='KEY_CODE_3')
fig = merge_and_plot_sig_coeffs(rdata_noout, summary_dict_vl, coef_vars=cols_reduced, ncols=2, main_title="Significant Coefficient Maps")

# FOR PCA
pca_boot_coeff =bootstrap_gwr_coefficients(pca_rdata_noout, cols_pca, 600, y_col='full')
fig = map_bootstrap_subplots(pca_rdata_noout, pca_boot_coeff, coef_vars=cols_pca, key_col='KEY_CODE_3', ncols=3)
summary_dict_pca = create_summary_dict(pca_boot_coeff, coef_vars=cols_pca, key_col='KEY_CODE_3')
fig = merge_and_plot_sig_coeffs(pca_rdata_noout, summary_dict_pca, coef_vars=cols_pca, ncols=3, main_title="Significant PCA coefficients: all parks")

# PCA very large
pca_boot_coeff =bootstrap_gwr_coefficients(pca_rdata_noout, cols_pca, 600, y_col='vl')
fig = map_bootstrap_subplots(pca_rdata_noout, pca_boot_coeff, coef_vars=cols_pca, key_col='KEY_CODE_3', ncols=3)
summary_dict_pca = create_summary_dict(pca_boot_coeff, coef_vars=cols_pca, key_col='KEY_CODE_3')
fig = merge_and_plot_sig_coeffs(pca_rdata_noout, summary_dict_pca, coef_vars=cols_pca, ncols=3, main_title="Significant PCA coefficients: largest parks")
