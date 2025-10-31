from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, kendalltau

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    max_error, mean_absolute_percentage_error, explained_variance_score
)

# Regressors
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, SGDRegressor, PassiveAggressiveRegressor
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Optional: XGBoost / LightGBM
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


def load_data(csv_path, target_column):
    """Load dataset and split features/target."""
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns: {df.columns.tolist()}")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def preprocess_data(X, y):
    """Encode categoricals, scale numericals, and split train/test."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    print(f"Data prepared for modeling. Final feature space: {X_train.shape[1]} features.\n")
    return X_train, X_test, y_train, y_test


def evaluate_and_plot(model, name, X_train, X_test, y_train, y_test, save_dir="model_plots"):
    """Train, evaluate, compute metrics, and plot predicted vs true."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Basic regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # Correlation metrics
    try:
        pearson_corr, _ = pearsonr(y_test, y_pred)
    except Exception:
        pearson_corr = np.nan
    try:
        kendall_corr, _ = kendalltau(y_test, y_pred)
    except Exception:
        kendall_corr = np.nan

    # Plot predicted vs true
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color="dodgerblue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title(f"{name} - Predicted vs True")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}_pred_vs_true.png", dpi=120)
    plt.close()

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "MaxError": max_err,
        "R2": r2,
        "ExplainedVar": evs,
        "Pearson": pearson_corr,
        "Kendall": kendall_corr
    }


def main(csv_path, target_column):
    print("Loading dataset...")
    X, y = load_data(csv_path, target_column)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.\n")

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    regressors = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "BayesianRidge": BayesianRidge(),
        "HuberRegressor": HuberRegressor(),
        "SGDRegressor": SGDRegressor(max_iter=1000, tol=1e-3),
        "PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=1000, tol=1e-3),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "ExtraTreesRegressor": ExtraTreesRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "AdaBoostRegressor": AdaBoostRegressor(),
        "BaggingRegressor": BaggingRegressor(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "SVR": SVR(),
        "MLPRegressor": MLPRegressor(max_iter=1000)
    }

    if XGB_AVAILABLE:
        regressors["XGBRegressor"] = XGBRegressor(objective='reg:squarederror', verbosity=0)
    if LGB_AVAILABLE:
        regressors["LGBMRegressor"] = LGBMRegressor()

    print("🔹 Training and evaluating models...\n")
    results = {}
    for name, model in tqdm(regressors.items()):
        try:
            results[name] = evaluate_and_plot(model, name, X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f" {name} failed: {e}\n")

    # Results summary
    print("\nSummary of all models:")
    df_results = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    print(df_results.round(4))

    # Save results
    df_results.to_csv("regression_model_comparison_results.csv")
    print("\nResults saved to 'regression_model_comparison_results.csv'")
    print("Plots saved to 'model_plots/' folder.")

    # R² bar chart
    plt.figure(figsize=(10, 6))
    df_results["R2"].plot(kind="barh", color="skyblue")
    plt.title("Model Comparison (R² Score)")
    plt.xlabel("R² Score")
    plt.tight_layout()
    plt.savefig("model_r2_comparison.png", dpi=120)
    plt.close()
    print("Summary R² comparison plot saved as 'model_r2_comparison.png'.")

    # Optional: heatmap of all metrics
    plt.figure(figsize=(12, 6))
    plt.imshow(df_results.corr(), cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(df_results.columns)), df_results.columns, rotation=45, ha="right")
    plt.yticks(range(len(df_results.columns)), df_results.columns)
    plt.title("Metrics Correlation Heatmap (Across Models)")
    plt.tight_layout()
    plt.savefig("metrics_correlation_heatmap.png", dpi=120)
    plt.close()
    print("Metrics correlation heatmap saved as 'metrics_correlation_heatmap.png'.")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare regression algorithms on a CSV dataset.")
    parser.add_argument("--csv", help="Path to the input CSV file.", default="../01_VFAs_dataset_imputed.csv")
    parser.add_argument("--target", help="Name of the target column.", default="VFAs")
    args = parser.parse_args()

    main(args.csv, args.target)
