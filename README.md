# Explaining Temporal Graph Predictions with Shapley Values

This is the repository for reproducing the findings in the Master thesis "Explaining Temporal Graph Predictions with Shapley Values". 

# Prerequisits

The implementation bases on Python 3.9.13 and CUDA 12.4. The external packages required to run the evaluation can be found in the [requirements.txt](/requirements.txt). To install the requirements execute `pip install -r /requirements.txt`. 

Additionally, execute 
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

# Content

## External Content

The repository contains code clones of the [DyGLib](https://github.com/yule-BUAA/DyGLib), [TGNNExplainer](https://github.com/cisaic/tgnnexplainer), and [TempME](https://github.com/Graph-and-Geometric-Learning/TempME), which are further adjusted to fit the needs. The clones can be found in [DyGLib](/DyGLib), [/Explainers/External/tgnnexplainer](/Explainers/External/tgnnexplainer), and [/Explainers/External/TempME](/Explainers/External/TempME).

## Contribution

The main contribution of the thesis can be found in [/Explainers/Shapley4TGNN](/Explainers/Shapley4TGNN), which contains the custom waterfall plot and both explainer versions (on event-level and on feature-level). Further, the explainer on feature-level implements two versions: One based on the two-step Owen values, and one based on hierarchy-conform permutations. 

# Evaluation

This section describes how to execute the explainer. Futher, it utilizes the MOOC dataset as an example.

Each dataset has its own folder in [/Evaluation](/Evaluation). Each folder contains at least:

- "training.py": Contains the logic for training the TGNN.
- "Statistics.ipynb": Delivers statistics about the dataset 
- "Quantitative Evaluation.ipynb": Evaluates the explainers

Some folders also contain a "Qualitative Evaluation.ipynb" which creates example explanations for the dataset. Further some folder contain additional folders that evaluate other configurations on the dataset. 

## 1. Add a dataset

The repository does not contain the data for the effect evaluation, and the data is not publicy available. 

All other datasets are available [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o). The data files need to be placed within a folder in [/Data](/Data). The available datasets mainly consist of four files: 

- `*.csv`: Original network, which is not needed for reproducing the findings. It needs to be deleted if you use a predefined configuration. 
- `ml_*.csv`: Network structure. If you use a predefined configuration, rename this file into `*.csv`. 
-  `ml_*.npy`: Event features. If you use a predefined configuration, rename this file into `*.npy`. 
- `ml_*_node.npy`: Node features, which are zero for the available datasets. If you use a predefined configuration, rename this file into `*_node.npy`.

### MOOC Example

Download the `mooc.zip` from [here](https://zenodo.org/records/7213796#.Y1cO6y8r30o). Unzip the files into [/Data/MOOC](/Data/MOOC), delete [/Data/MOOC/mooc.csv](/Data/MOOC/mooc.csv), rename the files according to the description above. To this end, the repository needs to contain the following files:
- [/Data/MOOC/mooc.csv](/Data/MOOC/mooc.csv)
- [/Data/MOOC/mooc.npy](/Data/MOOC/mooc.npy)
- [/Data/MOOC/mooc_node.npy](/Data/MOOC/mooc_node.npy)

## 2. Add a configuration

The configurations are located in [/Config](/Config) in form of `.yaml` files. To add a new configuration, duplicate an existing one and adjust it to your needs. The parameter descriptions can be found in [/Config/config.py](/Config/config.py). 

To make use of a configuration, execute 

```
from Config.config import CONFIG
CONFIG = CONFIG("FileNameWithout'.yaml'")
```

After that, you can use the configuration parameter, e.g. `CONFIG.data.folder` stores the location of the dataset. 

### MOOC example

The configuration for MOOC already exists in [/Config/MOOC.yaml](Config/MOOC.yaml). Each file, that exacutes

```
from Config.config import CONFIG
CONFIG = CONFIG("MOOC")
```

in the beginning, makes use of the MOOC configuration. 

## 3. Run training

The training is located in the training.py of the evaluation folder. It can be executed using 

```
python -m Evaluation.*.training
```

If another configuration used that is located in the evaluation folder, use 

```
python -m Evaluation.*.**.training
```

The resulting model is then located in [/Saved_models](/Saved_models). Each model gets its own folder specified by `CONFIG.data.dataset_name` following by the name of the TGNN used, e.g. "TGAT" or "TGN". The folder also contains performance metrics in form of ".json" files. 

### MOOC Example

Execute: 

```
python -m Evaluation.MOOC.training
```

The resulting model is then located in [/Saved_models/MOOC/TGAT/TGAT_seed0.pkl](/Saved_models/MOOC/TGAT/TGAT_seed0.pkl) and the performance metrics are in [/Saved_models/MOOC/TGAT/TGAT_seed0.json](/Saved_models/MOOC/TGAT/TGAT_seed0.json). 

For training the TGN on the MOOC dataset run:

```
python -m Evaluation.MOOC.MOOCTGN.training
```

The resulting model is then located in [/Saved_models/MOOCTGN/TGN/TGN_seed0.pkl](/Saved_models/MOOCTGN/TGN/TGN_seed0.pkl) and the performance metrics are in [/Saved_models/MOOCTGN/TGN/TGN_seed0.json](/Saved_models/MOOCTGN/TGN/TGN_seed0.json). 

## 4. Run Quantitative Evaluation

The evaluation is located in the "Quantitative Evaluation.py" of the evaluation folder. The evaluation is predefined to execute the preprocessing of TempME including motif extraction and training. This is done by setting `preprocessing = True`. It can be turned of if the preprocessing was executed before. Further, the number of sample explanations can be set with `num_samples`. The plots are saved in [/Documents/Images](/Documents/Images)

### MOOC Example

Navigate into [/Evaluation/MOOC/Quantitative Evaluation.ipynb](/Evaluation/MOOC/Quantitative%20Evaluation.ipynb). Execute all cells in order. After that, the results are stored in [Documents/Images/MOOC](Documents/Images/MOOC). The AUCs are directly stored in the notebook. 

## (Optional) 5. Run Qualitative Evaluation

The evaluation is located in the "Qualitative Evaluation.py" of the evaluation folder. It creates two samples using the Shapley explainer and stores them in [/Documents/Images](/Documents/Images).

### MOOC Example

Navigate into [/Evaluation/MOOC/Qualitative Evaluation.ipynb](/Evaluation/MOOC/Qualitative%20Evaluation.ipynb). Execute all cells in order. After that, the results are stored in [Documents/Images/MOOC](Documents/Images/MOOC).



