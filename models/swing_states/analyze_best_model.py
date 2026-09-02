from pathlib import Path
import pandas as pd

from train_models import (
    load_demographics,
    aggregate_demographics,
    prepare_features,
    select_features,
    get_top_features_initial,
    train_random_forest,
    analyze_shap_values,
    analyze_swing_states,
    plot_difference_lollipop,
    save_predictions,
    TOP_N_FEATURES,
    OUTPUT_DIR
)

# Features to exclude from SHAP interpretation to avoid circularity
ELECTORAL_SHARE_FEATURES = [
    "elec_top1_share",
    "elec_top2_share",
    "elec_top2_margin",
    "elec_herfindahl_index",
    "elec_effective_num_parties",
]

def main():
    # Load best method
    with open(OUTPUT_DIR / "best_method.txt") as f:
        best_method = f.read().strip()

    method_dir = Path("processed-data/features/swing_states") / best_method
    election_df = pd.read_csv(method_dir / "election_features_full.csv")
    temporal_df = pd.read_csv(method_dir / "temporal_features.csv")

    demo_df = load_demographics()
    demo_agg = aggregate_demographics(demo_df) if not demo_df.empty else pd.DataFrame()

    # Prepare features
    df = prepare_features(election_df, temporal_df, demo_agg)

    X_all, y, feature_cols = select_features(df)

    # Feature selection
    top_features, _ = get_top_features_initial(X_all, y, TOP_N_FEATURES)
    X = X_all[top_features]

    # Train final model
    rf, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, y_train_pred, _ = (
        train_random_forest(X, y)
    )

    # Interpretation feature set (remove electoral share features)
    shap_features = [
    f for f in top_features if f not in ELECTORAL_SHARE_FEATURES
    ]

    X_shap_train = X_train[shap_features]
    X_shap_test = X_test[shap_features]

    # Train interpretation-only model
    rf_shap, _, _, _, _, _, _, _, _ = train_random_forest(
        X_all[shap_features], y
    )

    # SHAP (interpretation model)
    shap_importance, shap_values = analyze_shap_values(
        rf_shap,
        X_shap_train,
        X_shap_test,
        shap_features
    )

    # Swing comparison
    comparison_df = analyze_swing_states(df, top_features)

    # Predictions
    save_predictions(df, rf, X)

if __name__ == "__main__":
    main()