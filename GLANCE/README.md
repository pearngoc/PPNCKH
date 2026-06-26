# GLANCE: Global Actions in a Nutshell for Counterfactual Explainability

This is the home directory of our GLANCE framework. Before trying to run our project, please consult the setup instructions below.

## Table of Contents
- [Structure](#structure)
- [Setup Instructions](#setup-instructions)
- [Example Notebooks](#example-notebooks)

## Structure
    .
    ├── examples                  # Folder containing example notebooks and relevant files
    │   ├── datasets              # Folder containing the datasets used in our examples  
    │   |    ├── adult.data            
    │   |    ├── compas.data  
    |   |    ├── default.data  
    |   |    ├── german.data  
    |   |    ├── heloc.data   
    │   ├── models              # Folder containing the trained models used in our examples  
    │   |    ├── compas_lr.pkl            
    │   |    ├── compas_xgb.pkl  
    |   |    ├── compas_dnn.pkl  
    |   |    └── ...   
    │   ├── Adult.ipynb
    │   ├── Compas.ipynb
    │   ├── DefaultCredit.ipynb
    │   ├── GermanCredit.ipynb
    │   ├── Heloc.ipytnb              
    ├── src                     # Source code folder
    │   ├── glance     
    │   │   ├── glance              # Contains the code of GLANCE framework
    │   │   ├── clustering          # Contains the code for the clustering method
    │   │   ├── utils               # Contains helper functions
    │   │   └── local_cfs           # Contains the code for the local counterfactual methods
    └── ...                     # README, requirements.txt, pyproject.toml

## Setup Instructions

### Installation Steps

1. **Create a Virtual Environment:**
   
    Preferable, create a virtual environment with python==3.10.4 and activate it. 

    Using Conda:
    ```bash
    conda create --name glance-aaai python==3.10.4
    conda activate glance-aaai
    ```

    Using python venv:
    ### Prerequisites
    
    Make sure you have the following prerequisites installed:
    - **Python** (version 3.10.4)
    - **pip**
    ```bash
    python -m venv glance-aaai
    glance-aaai/bin/activate  
    ```


3. **Install Dependencies:**
   
    Use the file `requirements.txt` to install all dependencies of our framework:

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

4. **Set Up Jupyter Notebook (Optional for Example Notebooks):**
   
    Additionally, if you wish to run the example notebooks (described below) you should run the following commands to appropriately setup and start the `jupyter notebook` application:

    ```bash
    python -m ipykernel install --user --name=glance-aaai --display-name "glance env"
    jupyter notebook

    ```

## Example Notebooks

As a gateway for the user, there exist 4 example notebooks in the `examples/` directory:

- `COMPAS.ipynb`
- `DefaultCredit.ipynb`
- `GermanCredit.ipynb`
- `HELOC.ipynb`
- `Adult.ipynb`

Upon successful completion of the setup, the notebooks should execute as intended. They demonstrate basic usage of our framework, allowing users to adjust parameter values as needed.

## Running Experiments

To run the large-scale experiments that were showcased in the paper, use the `script.py` file located at the root of the project. This script reads a parameter configuration file (`param_grid.txt`) and executes experiments accordingly.

### Step-by-Step:

1. **Prepare the configuration file**

   Create a txt file (e.g., `param_grid.txt`) with the parameter grid and any excluded combinations. Example:

   ```json
   {
     "param_grid": {
       "dataset": ["adult"],
       "model": ["xgb"],
       "method": ["GLANCE"],
       "local_cf_generator": ["Dice"],
       "clustering_method": ["KMeans"],
       "n_initial_clusters": [100],
       "n_final_clusters": [4],
       "n_local_counterfactuals": [10],
       "IM__cluster_action_choice_algo": ["max-eff"]
     },
     "exclude_combinations": []
   }

2. **Run the experiment script**
```bash
python script.py --input param_grid.txt --output results.csv --error errors.log
```

### Reproducing the Experiments
All experiments reported in the appendix of the paper were conducted by configuring this param_grid.txt file.
- To reproduce experiments comparing different local counterfactual generation methods, modify the local_cf_generator field to include options like: 
**["Dice", "NearestNeighbors", "NearestNeighborsScaled", "RandomSampling"]**
- To reproduce experiments comparing clustering strategies, modify the clustering_method field to include: **["KMeans", "Agglomerative", "GMM"]**
- To explore different numbers of clusters or counterfactuals, adjust the values of **n_initial_clusters**, **n_final_clusters**, or **n_local_counterfactuals** accordingly.
- To explore the experiments considering the local counterfactual method and how its costs compare to the global method, modify the method field and add **["Local"]**.
