import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Optional imports
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

    print(f"Found {len(cat_cols)} categorical columns and {len(num_cols)} numeric columns.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test


def evaluate_and_plot(model, name, X_train, X_test, y_train, y_test, save_dir="model_plots"):
    """Train, evaluate, and plot predicted vs. true values."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    #
    # print(f"********{name} *******")
    # print(f"   RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}\n")

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

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


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

    print("Training and evaluating models...\n")
    results = {}
    for name, model in tqdm(regressors.items()):
        try:
            results[name] = evaluate_and_plot(model, name, X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f" {name} failed: {e}\n")

    # Results summary
    print("\n Summary of all models:")
    df_results = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    print(df_results.round(4))

    df_results.to_csv("regression_model_comparison_results.csv")
    print("\n Results saved to 'regression_model_comparison_results.csv'")
    print("Plots saved to 'model_plots/' folder.")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare regression algorithms on a CSV dataset.")
    parser.add_argument("--csv", help="Path to the input CSV file.", default="../01_VFAs_dataset_imputed.csv")
    parser.add_argument("--target", help="Name of the target column.", default="VFAs")
    args = parser.parse_args()

    main(args.csv, args.target)
