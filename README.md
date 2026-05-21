# TGNNSHAP

This is the repository for reproducing the findings in the paper "TGNNSHAP: A Hierarchical Explainer for Temporal Graph Predictions using Shapley and Owen Values". 

# Prerequisites

The implementation is based on Python 3.9.13 and CUDA 12.4. The external packages required to run the evaluation can be found in the [requirements.txt](/requirements.txt). To install the requirements, execute `pip install -r requirements.txt`. 

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

This section describes examplatory how to execute the explainer using the artificial dataset. Futher, this description includes how to generate the data.

## 1. Add the dataset

To generate the artificial dataset execute 

```python -m Evaluation.Generated.generate_data```

This command creates the dataset including a network structure file **ml_generated.csv**, event features **ml_generated.npy**, and node features **ml_generated_node.npy**. All files are stored in [/Data/generated](/Data/generated).

All other datasets are available [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o). The data files need to be placed within a folder in [/Data](/Data). The available datasets mainly consist of four files: 

- `*.csv`: Original network, which is not needed for reproducing the findings. 
- `ml_*.csv`: Network structure. 
-  `ml_*.npy`: Event features. 
- `ml_*_node.npy`: Node features, which are zero for the available datasets. 

### MOOC Example

Download the `mooc.zip` from [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o). Unzip the files into [/Data/mooc](/Data/mooc). To this end, the repository needs to contain the following files:
- [/Data/mooc/mooc.csv](/Data/mooc/mooc.csv)
- [/Data/mooc/ml_mooc.csv](/Data/mooc/ml_mooc.csv)
- [/Data/mooc/ml_mooc.npy](/Data/mooc/ml_mooc.npy)
- [/Data/mooc/ml_mooc_node.npy](/Data/mooc/ml_mooc_node.npy)

## 2. Add a configuration

The configurations are located in [/Config](/Config) in the form of `.yaml` files. In case of reproduction, there are not futher actions to do in this step. 

The parameter descriptions of all configurations can be found in [/Config/config.py](/Config/config.py). 

To make use of the configuration for the artificial dataset, execute 

```
from Config.config import CONFIG
CONFIG = CONFIG("Generated")
```

This loads the configuration with the name `Generated.yaml`. After that, you can use the configuration parameters that are stored in [Generated.yaml](/Config/Generated.yaml). E.g., `CONFIG.data.folder` stores the location of the dataset. 

## 3. Run training

The training routines are located [Evaluation/Training](Evaluation/Training). For regression tasks e.g. on the artificial dataset use  

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

```python -m Evaluation.run_eval --dataset "Generated" --explainer all```

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

To visualize the results one can use [Evaluation/visualize.ipynb](Evaluation/visualize.ipynb). There, the exemplaratory implementation for the Reddit dataset is given. To use another dataset, change the configuration used (first cell). If the notebook has been executed before, restart it to clear the cached configuration.

The plots are stored in [Documents/Images/Reddit](Documents/Images/Reddit). The innermost folder depends on the selected configuration and changes if the used configuration is changed.



