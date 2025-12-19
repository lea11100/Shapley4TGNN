# Explaining Temporal Graph Predictions with Shapley Values

This is the repository for reproducing the findings in the Master thesis "Explaining Temporal Graph Predictions with Shapley Values". 

# Prerequisites

The implementation is based on Python 3.9.13 and CUDA 12.4. The external packages required to run the evaluation can be found in the [requirements.txt](/requirements.txt). To install the requirements, execute `pip install -r /requirements.txt`. 

Additionally, execute 
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

# Content

## External Content

The repository contains code clones of the [DyGLib](https://github.com/yule-BUAA/DyGLib), [TGNNExplainer](https://github.com/cisaic/tgnnexplainer), and [TempME](https://github.com/Graph-and-Geometric-Learning/TempME), which are further adjusted to fit the needs. The clones can be found in [DyGLib](/DyGLib), [/Explainers/External/tgnnexplainer](/Explainers/External/tgnnexplainer), and [/Explainers/External/TempME](/Explainers/External/TempME).

## Contribution

The main contribution of the thesis can be found in [/Explainers/Shapley4TGNN](/Explainers/Shapley4TGNN), which contains the custom waterfall plot and both explainer versions (on event-level and on feature-level). Further, the explainer on feature-level implements two versions: One based on the two-step Owen values and one based on hierarchy-conform permutations. 

# Evaluation

This section describes examplatory how to execute the explainer using the artificial dataset. Futher this description includes how to generate the data.

## 1. Add the dataset

To generate the artificial dataset execute 

```python -m Evaluation.Generated.generate_data```

This command creates the dataset including a network structure file **Generated.csv**, event features **Generated.npy**, and node features **Generated_node.npy**. All files are stored in [/Data/Generated](/Data/Generated).

All other datasets are available [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o). The data files need to be placed within a folder in [/Data](/Data). The available datasets mainly consist of four files: 

- `*.csv`: Original network, which is not needed for reproducing the findings. It needs to be deleted if you use a predefined configuration. 
- `ml_*.csv`: Network structure. If you use a predefined configuration, rename this file into `*.csv`. 
-  `ml_*.npy`: Event features. If you use a predefined configuration, rename this file into `*.npy`. 
- `ml_*_node.npy`: Node features, which are zero for the available datasets. If you use a predefined configuration, rename this file to `*_node.npy`.

### MOOC Example

Download the `mooc.zip` from [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o). Unzip the files into [/Data/MOOC](/Data/MOOC), delete [/Data/MOOC/mooc.csv](/Data/MOOC/mooc.csv), and rename the files according to the description above. To this end, the repository needs to contain the following files:
- [/Data/MOOC/mooc.csv](/Data/MOOC/mooc.csv)
- [/Data/MOOC/mooc.npy](/Data/MOOC/mooc.npy)
- [/Data/MOOC/mooc_node.npy](/Data/MOOC/mooc_node.npy)

## 2. Add a configuration

The configurations are located in [/Config](/Config) in the form of `.yaml` files. The parameter descriptions can be found in [/Config/config.py](/Config/config.py). 

To make use of the configuration for the artificial dataset, execute 

```
from Config.config import CONFIG
CONFIG = CONFIG("Generated")
```

After that, you can use the configuration parameters that are stored in [Generated.yaml](/Config/Generated.yaml). E.g., `CONFIG.data.folder` stores the location of the dataset. 

## 3. Run training

The trainig routines are located [Evaluation/Training](Evaluation/Training). For regression tasks e.g. on the artificial dataset use  

```
python -m Evaluation.Training.regression -d "Generated"
```

For prediction tasks use 

```
python -m Evaluation.Training.prediction -d "MOOC"
```

The flag ``-d`` specifies the configuration used.

The resulting model is then located in [/Saved_models](/Saved_models). Each model gets its own folder specified by `CONFIG.data.dataset_name` followed by the name of the TGNN used, e.g., "TGAT" or "TGN". The folder also contains performance metrics in the form of ".json" files. 

## 4. Run Quantitative Evaluation

To evaluate the explainer performances for the artificial dataset use

```python -m Evaluation.run_exec --dataset "Generated" --explainer all```

This command evaluates the the baseline explainers and the novel Shapley explainers. The results are stored in [Documents/ExplainerOutputs/](Documents/ExplainerOutputs/) where each explainer receives its own CSV file. 

The evaluation supports the following parameters:

### Required Arguments

- `-d, --dataset DATASET` -
  Name of the dataset/configuration to use.  

- `--explainer EXPLAINER` -
  Name of the explainer method to use. Supported values: "shapley_event", "shapley_feature", "tgnn", "tempme", "all"

### Optional Arguments

- `--preprocessing` (default: `True`) -
  Whether to apply preprocessing (motif extraction and training) for TempME before evaluation.  
  Set to `False` to disable preprocessing.

- `--num_samples NUM_SAMPLES` (default: `200`) - 
  Number of samples to use during evaluation.

- `--store_coalitions` (default: `False`) -
  Whether to store coalitions generated during evaluation.

## 5. Visualize results

To visualize the results one can use [Evaluation/visualize.ipynb](Evaluation/visualize.ipynb). There, the exemplaratory implementation for the Reddit dataset is given. To use another dataset, change the configuration used (first cell) and load other CSVs (second cell).

The plots are stored in [Documents/Images/Reddit](Documents/Images/Reddit). The innermost folder depends on the selected configuration and changes if the used configuration is changed.

## (Optional) 5. Run Qualitative Evaluation

[Evaluation/Generated/Qualitative Evaluation.ipynb](Evaluation/Generated/Qualitative%20Evaluation.ipynb) presents an exemplatory usage of the hierachical waterfall diagramm to visualize explanations for the artificial dataset. 

The plot are then stored in [Documents/Images/Generated](Documents/Images/Generated).



