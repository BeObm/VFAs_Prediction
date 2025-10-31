import os
import torch
import torch.nn as nn
import pandas as pd
import  random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
from utils import *
import warnings
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    # os.CUBLAS_WORKSPACE_CONFIG="4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.iloc[:, 2:]
    df.columns = df.columns.str.strip()
    if 'Codigestion' in df.columns:
        df['Codigestion'] = LabelEncoder().fit_transform(df['Codigestion'])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X[X.columns] = StandardScaler().fit_transform(X[X.columns])
    return X, y


class MyDataset(Dataset):
    def __init__(self, X, y,type_model):
        if type_model == 'cnn':
            self.X = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)  # CNN expects input shape: (B, 1, F)
        elif type_model in ['mlp',"deep_mlp"]:
            self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ResidualBlock(nn.Module):
    def __init__(self, features, dropout=0.2, stochastic_depth_prob=0.0):
        super().__init__()
        self.stochastic_depth_prob = stochastic_depth_prob
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.BatchNorm1d(features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(features, features),
            nn.BatchNorm1d(features)
        )
        self.relu = nn.GELU()

    def forward(self, x):
        if self.training and random.random() < self.stochastic_depth_prob:
            return x  # skip block (stochastic depth)
        return self.relu(x + self.block(x))  # residual connection


class DeepMLPRegressor(nn.Module):
    def __init__(self, in_features, hidden_features=128, num_layers=100, dropout=0.2,
                 stochastic_depth_prob=0.1, use_input_noise=True, noise_std=0.01):
        super().__init__()

        self.use_input_noise = use_input_noise
        self.noise_std = noise_std

        self.input_layer = nn.Linear(in_features, hidden_features)
        self.input_bn = nn.BatchNorm1d(hidden_features)
        self.relu = nn.GELU()

        # Residual blocks with optional stochastic depth
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_features, dropout, stochastic_depth_prob) for _ in range(num_layers)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_features, 1)

    def forward(self, x):
        if self.training and self.use_input_noise:
            x = x + torch.randn_like(x) * self.noise_std

        x = self.relu(self.input_bn(self.input_layer(x)))
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x


class MLPRegressor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x): return self.net(x)



import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))  # Residual connection

class CNNRegressor(nn.Module):
    def __init__(self, input_features, num_blocks=200, start_channels=1024):
        super().__init__()
        self.input_layer = nn.Conv1d(1, start_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Stack multiple residual blocks
        layers = []
        channels = start_channels
        for i in range(num_blocks):
            layers.append(ResidualBlock1D(channels))
            # Optionally reduce channels every few blocks
            if (i + 1) % 20 == 0 and channels > 16:
                layers.append(nn.Conv1d(channels, channels // 2, kernel_size=1))
                layers.append(nn.ReLU())
                channels = channels // 2

        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, 1)
        )

    def forward(self, x):  # x: (B, 1, F)
        x = self.relu(self.input_layer(x))
        x = self.cnn(x)
        x = self.pool(x)
        return self.regressor(x)



#
# class CNNRegressor(nn.Module):
#     def __init__(self, input_features):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv1d(1, 1024, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.Conv1d(1024, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.Conv1d(512, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#             nn.Conv1d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#
#
#             nn.Conv1d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1),
#         )
#         self.regressor = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x):  # x: (B, 1, F)
#         x = self.cnn(x)
#         return self.regressor(x)



def train_and_evaluate(args, model, train_loader, val_loader, name=""):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))

        print(f"[{name}] Epoch {epoch+1:02d} | Train MSE: {train_losses[-1]:.4f} | Val MSE: {val_losses[-1]:.4f}")

    return train_losses, val_losses



def get_predictions(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            preds.extend(pred)
            targets.extend(yb.numpy())
    return np.array(preds).flatten(), np.array(targets).flatten()


def compute_shap_importance(
        args,
        model,
        X_sample,
        feature_names,
        threshold,
        model_type):
    model.eval()
    input_tensor = torch.tensor(X_sample.values, dtype=torch.float32).to(DEVICE)

    # Reshape if CNN
    if model_type == "cnn":
        input_tensor = input_tensor.unsqueeze(1)  # Shape: (B, 1, F)

    # Use background of first 100 samples
    background = input_tensor[:100]

    # === Choose SHAP Explainer ===
    if model_type == "cnn":
        explainer = shap.DeepExplainer(model, background)
    else:  # Assume MLP
        explainer = shap.GradientExplainer(model, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(input_tensor)

    # If model output is single-class (binary), take the first array
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Reshape if needed (especially for CNNs)
    if model_type == "cnn" and shap_values.ndim == 3:
        shap_values = shap_values.squeeze(1)

    # === Compute mean absolute SHAP and convert to percentage ===
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    percent_importance = (mean_abs_shap / mean_abs_shap.sum()) * 100
    all_importance = pd.Series(percent_importance, index=feature_names).sort_values(ascending=False)

    # === Select features above threshold ===
    selected_features = all_importance[all_importance > threshold]

    # === Save to Excel ===
    excel_path = os.path.join(args.output_dir, f"shap_feature_importance_{args.type_model}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        all_importance.to_frame("SHAP (%)").to_excel(writer, sheet_name="All Features")

    # === Plot ===
    plot_path = os.path.join(args.output_dir, f"Shap_feature_importance_{args.type_model}.pdf")
    plt.figure(figsize=(12, 6))
    all_importance.plot(kind="bar", color="teal", edgecolor="black")
    plt.ylabel("Mean SHAP Importance (%)")
    plt.title("Feature Importance (Percentage)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path, format="pdf")
    plt.close()

    return selected_features, all_importance, shap_values

def plot_feature_importance(output_dir,importance: pd.Series, title, filename):
    plt.figure(figsize=(10, 6))
    importance.sort_values().plot(kind='barh', color='teal')
    plt.title(title)
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_losses(output_dir,train_losses, val_losses, title="Loss Curve", filename="loss_curve.pdf"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# def plot_preds(p, t, dataset_name, model_name):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(t, p, alpha=0.6, edgecolors='w', linewidth=0.5)
#     plt.plot([t.min(), t.max()], [t.min(), t.max()], 'r--', linewidth=2)
#     plt.xlabel("True Values")
#     plt.ylabel("Predicted Values")
#     plt.title(f"{model_name} - {dataset_name} Predicted vs True")
#     plt.grid(True)
#     plt.tight_layout()
#     fname = f"{model_name.lower().replace(' ', '_')}_{dataset_name.lower()}_pred_vs_true.pdf"
#     plt.savefig(os.path.join(output_dir, fname))
#     plt.close()


def print_and_plot_preds(args,model, train_loader, val_loader, name="Model"):
    preds_train, targets_train = get_predictions(model, train_loader)
    preds_val, targets_val = get_predictions(model, val_loader)

    df_preds = pd.DataFrame({
        "True": targets_val,
        "Predicted": preds_val
    })
    excel_path = os.path.join(args.output_dir, f"{name.lower()}_predictions.xlsx")
    df_preds.to_excel(excel_path, index=False)

    def print_metrics(p, t, dataset_name):
        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mae = mean_absolute_error(t, p)
        print(f"\n{name} - {dataset_name} Metrics:")
        print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        return  {"R2": r2, "RMSE": rmse, "MAE": mae}

    m_train = print_metrics(preds_train, targets_train, "Train")
    m_val = print_metrics(preds_val, targets_val, "Validation")


    def plot_preds(p, t, dataset_name):
        plt.figure(figsize=(6, 6))
        plt.scatter(t, p, alpha=0.6, edgecolors='w', linewidth=0.5)
        plt.plot([t.min(), t.max()], [t.min(), t.max()], 'r--', linewidth=2)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{name} - {dataset_name} Predicted vs True")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plot_preds(preds_train, targets_train, "Train")
    plot_preds(preds_val, targets_val, "Validation")

    return m_train, m_val




def save_metrics_summary(args,metrics_dict):
    filename= args.sumary_filename

    df = pd.DataFrame(metrics_dict).T
    df.index.name = "Set"
    df.insert(0, "Model", args.type_model.upper())

    path = os.path.join(args.output_dir, filename)

    if filename.endswith(".xlsx"):
        df.to_excel(path)
    elif filename.endswith(".csv"):
        df.to_csv(path)
    elif filename.endswith(".txt"):
        with open(path, "w") as f:
            f.write(df.to_string())
    else:
        raise ValueError("Unsupported file format. Use .xlsx, .csv or .txt")

