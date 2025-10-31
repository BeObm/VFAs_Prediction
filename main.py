import argparse
from utils import *
import warnings
warnings.filterwarnings("ignore")



def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    X, y = load_data(args.dataset)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    train_loader = DataLoader(MyDataset(X_train, y_train, type_model=args.type_model), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MyDataset(X_val, y_val,type_model=args.type_model), batch_size=args.batch_size)
    if args.type_model == "mlp":
        model_full = MLPRegressor(in_features=X.shape[1])
    elif args.type_model == "deep_mlp":
        model_full = DeepMLPRegressor(in_features=X.shape[1])
    elif args.type_model == "cnn":
        model_full = CNNRegressor(input_features=X.shape[1])

    print(f"{'*'*10}  Training {args.type_model} model on all features...")

    losses_train_full, losses_val_full = train_and_evaluate(args,model_full, train_loader, val_loader, name="All_features")
    m_train_full, m_val_full = print_and_plot_preds(args,model_full, train_loader, val_loader, name=f"Full {args.type_model} Model")
    plot_losses(args.output_dir,losses_train_full, losses_val_full, title=f"Loss (All Features-{args.type_model} Model) ")

    print(f"{'*'*10} Computing SHAP Feature Importance...")
    top_features, _,_ = compute_shap_importance(args,model_full, X_train, X.columns, threshold=args.threshold,model_type=args.type_model)
    print(f"Top Features for {args.type_model} Model:", top_features.index.tolist())
    plot_feature_importance(args.output_dir,top_features, f"Top Features ({args.type_model} Model)", filename=f"shap_feature_importance_{args.type_model}-Model.pdf" )


    X_top = X[top_features.index]
    X_train_top, X_val_top, y_train, y_val = train_test_split(X_top, y, test_size=0.2, random_state=42)
    train_loader_top = DataLoader(MyDataset(X_train_top, y_train,type_model=args.type_model), batch_size=args.batch_size, shuffle=True)
    val_loader_top = DataLoader(MyDataset(X_val_top, y_val,type_model=args.type_model), batch_size=args.batch_size)

    if args.type_model == "mlp":
        model_top = MLPRegressor(in_features=X_top.shape[1])
    elif args.type_model == "deep_mlp":
        model_top = DeepMLPRegressor(in_features=X_top.shape[1])
    elif args.type_model == "cnn":
        model_top = CNNRegressor(input_features=X_top.shape[1])


    print(" {'*'*10}  Retraining with top features only...")

    losses_train_top, losses_val_top = train_and_evaluate(args,model_top, train_loader_top, val_loader_top, name="Top")
    m_train_top, m_val_top = print_and_plot_preds(args,model_top, train_loader_top, val_loader_top, name=f"Top_{args.type_model}_Features")
    plot_losses(args.output_dir,losses_train_top, losses_val_top, title=f"Loss (Top Features-{args.type_model} Model)")

    plot_losses(args.output_dir,losses_train_full, losses_val_full, title=f"Loss (All Features-{args.type_model} Model)", filename=f"loss_all_features_{args.type_model}-Model.pdf")
    plot_feature_importance(args.output_dir,top_features, f"Top Features (SHAP)",
                            filename=f"shap_feature_importance_{args.type_model}-Model.pdf")
    plot_losses(args.output_dir,losses_train_top, losses_val_top, title=f"Loss (Top Features)",
                filename=f"loss_top_features_{args.type_model}-Model.pdf")


    metrics_dict = {
        "Full_Train": m_train_full,
        "Full_Val": m_val_full,
        "Top_Train": m_train_top,
        "Top_Val": m_val_top,
    }

    save_metrics_summary(args,metrics_dict)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="01_VFAs_dataset_imputed.csv", help="path to dataset file")
    parser.add_argument("--type_model", default="deep_mlp", help="type of model", choices=["mlp", "cnn","deep_mlp"])
    parser.add_argument("--batch_size", default=64, help="Batch size")
    parser.add_argument("--epochs", default=500, help="max epochs")
    parser.add_argument("--test_size", default=0.2, help="test split size [0,1]")
    parser.add_argument("--threshold", default=0.3, help="threshold [0,1]")
    parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--seed", default=42, help="seed for reproducibility")
    parser.add_argument("--sumary_filename", default="result_sumary.xlsx", help="path to result summary file")

    args = parser.parse_args()
    args.output_dir = f"{args.type_model}-Output"
    main(args)

