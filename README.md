# Tokyo UGS Accessibility
This repository contains the code used for my Master's thesis. <br>

## Repo structure
The repository is organized as follows: 
- `notebooks` folder, which contains jupyter notebooks where some processes are explained in detail. Specifically, there are two notebooks:
    - `park_layer`, which lays out the process to obtain the urban green spaces layer
    - `e2sfca`, which explains how the accessibility index was calculated
- `scripts` folder, which contains `.py` files to carry out specific operations.
    - `accessibility_functions.py`: contains many functions necessary in the computation of the accessibility index
    - `analysis.py`: after the e2sfca index is obtained, it performs statistical analyses on the final data
    - `clean_accesses.py`: after obtaining the UGS layer and computing the accesses to UGS with QGIS, this checks for irregularities in the access points and census centroids.
    - `e2sfca.py`: performs the e2sfca method to obtain the accessibility metric. A commented version is found under 'notebooks'
    - `downlaod_data.py` allows downloading the data necessary to run the code.
- `environment.yml` allows to install all the necessary dependencies for the project

## How to use the repository
1. Install the relevant dependencies by using the `environment.yml` file.
2. Download the data at [the following link](https://drive.google.com/file/d/1mYUsnk-HIr2ES5S4Dfy1gjC_G_3Ex-h6/view?usp=sharing)
3. Place the "data" folder in the cloned repository. After extracting the folder "data" from the link above, the scripts should work using the relative path.
4. Replicate the processes
   - To evaluate the accessibility estimation, the `e2sfca.ipynb` can be checked. The code on the notebook can be interacted with, especially when it comes to visualizations.
   - To focus on the final analysis, the `analysis.py` file should be considered.  
