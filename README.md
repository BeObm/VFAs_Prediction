# VFAs Prediction Project

This project trains and evaluates MLP (Multi-Layer Perceptron) or CNN (Convolutional Neural Network) regression models on VFA dataset, analyzes feature importance using SHAP, and compares model performance when trained on all features vs. top features only.

It automatically:
1. Loads your dataset  
2. Trains the model  
3. Evaluates it on a validation set  
4. Computes SHAP feature importance  
5. Retrains the model using only top features  
6. Saves all results, plots, and summary metrics  


## How to use this code

Before running the code, you need to have Python 3.8 or newer installed.



## Step 1 — Install the Required Libraries

Open a **terminal (or command prompt)** in the same folder and run:

```bash
pip install -r requirements.txt
```

This command installs all the Python packages needed to run this code

## Step 2- Dataset


If you use your own data, just replace the file `01_VFAs_dataset_imputed.csv` with your dataset file.
Important: Make sure your dataset file has the same columns and if it does not you need to modify utils.load_data() accordingly

## Step 3 — Run the Script

To start training, run the following command in your terminal:

```bash
python main.py
```

By default, this will:
- Use the file `01_VFAs_dataset_imputed.csv`
- Train an MLP model
- Save all results in a folder named `MLP-Output`


## Step 4 — Optional Arguments (Customization)

You can customize how the script runs using **arguments**.  
Here are the main options you can change:

| Argument | Description | Default | Example |
|-----------|-------------|----------|----------|
| `--dataset` | Path to your dataset CSV file | `01_VFAs_dataset_imputed.csv` | `--dataset my_data.csv` |
| `--type_model` | Model type: `mlp` or `cnn` | `mlp` | `--type_model cnn` |
| `--output_dir` | Folder to save results | `MLP-Output` | `--output_dir results_cnn` |
| `--batch_size` | Training batch size | `64` | `--batch_size 128` |
| `--epochs` | Number of training epochs | `500` | `--epochs 300` |
| `--test_size` | Portion of data used for validation | `0.2` | `--test_size 0.3` |
| `--threshold` | SHAP feature importance threshold | `0.3` | `--threshold 0.25` |
| `--lr` | Learning rate | `0.01` | `--lr 0.001` |
| `--seed` | Random seed (for reproducibility) | `42` | `--seed 123` |
| `--sumary_filename` | Output Excel file for metrics summary | `result_sumary.xlsx` | `--sumary_filename results.xlsx` |

### Example — Training a CNN Model
```bash
python main.py --type_model cnn --epochs 300 --dataset my_data.csv --output_dir CNN_Results
```

---

## Outputs 

After the script runs, you will find everything inside the folder you set in `--output_dir` (default is `MLP-Output`):

| File/Folder | Description |
|--------------|--------------|
| `loss_all_features_<model>.pdf` | Loss curve for training with all features |
| `loss_top_features_<model>.pdf` | Loss curve for training with top features |
| `shap_feature_importance_<model>.pdf` | SHAP feature importance bar chart |
| `result_sumary.xlsx` | Excel file summarizing model metrics |
| `*.png` plots | Prediction vs. Ground Truth comparison |


## Example Folder Structure After Running

```
project_folder/
│
├── main.py
├── utils.py
├── requirements.txt
├── 01_VFAs_dataset_imputed.csv
│
└── MLP-Output/
    ├── loss_all_features_mlp-Model.pdf
    ├── loss_top_features_mlp-Model.pdf
    ├── shap_feature_importance_mlp-Model.pdf
    ├── result_sumary.xlsx
    └── other plots...
```


## Troubleshooting

| Issue | Possible Fix |
|--------|---------------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError` | Check that the dataset path is correct |
| Training is too slow | Lower `--epochs` or reduce dataset size |
| Plots not appearing | Check your output directory for saved `.pdf` files |

-
