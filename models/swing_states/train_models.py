import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.rcfonts": False,      # don't override LaTeX fonts
    "text.usetex": True,       # use LaTeX for all text
    "font.family": "serif",    # match LaTeX document
    "font.size": 20,           # readable size
})

# Paths
DEMOGRAPHICS_DIR = Path("processed-data/demographics")
OUTPUT_DIR = Path("models/swing_states/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2009, 2013, 2017, 2021]
JOIN_KEY = 'Gruppe'
TOP_N_FEATURES = 20  # Use only top 20 features

def load_demographics(years=YEARS):
    dfs = []
    for year in years:
        file_path = DEMOGRAPHICS_DIR / f"relative_{year}.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        df = pd.read_csv(file_path)
        df['year'] = year
        
        if 'Gruppe' not in df.columns:
            print(f"Warning: Gruppe column not found in {file_path}")
            continue
        
        df['Gruppe'] = df['Gruppe'].astype(str)
        
        # Remove Homes columns, keep People columns
        homes_cols = [col for col in df.columns if 'Homes' in col]
        df = df.drop(columns=homes_cols)
        print(f"Year {year}: Removed {len(homes_cols)} 'Homes' columns")
        
        cols_to_drop = ['ValgstedId', 'Valgsted navn', 'KredsNr', 'Kreds navn', 'Kommune navn']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# Merge several demographic features into a single feature set
def merge_features(df):
    mappings = {
        'Long_higher_education': [
            'demo_Education_18-29 years_Long higher education',
            'demo_Education_30-70+ years_Long higher education'
        ],
        'Vocational_training': [
            'demo_Education_18-29 years_Vocational training',
            'demo_Education_30-70+ years_Vocational training'
        ],
        'Primary_and_lower_secondary': [
            'demo_Education_18-29 years_Primary and lower secondary',
            'demo_Education_30-70+ years_Primary and lower secondary'
        ],
        'Not_specified': [
            'demo_Education_18-29 years_Not specified',
            'demo_Education_30-70+ years_Not specified'
        ],
        'Total_education': [  
            'demo_Education_Total 18-29 years',
            'demo_Education_Total 30-70+ years'
        ],
        'Age_0-17': [
            'demo_Age_0-4 years',
            'demo_Age_5-9 years',
            'demo_Age_10-14 years',
            'demo_Age_15-17 years'
        ],
        'Age_18-39': [
            'demo_Age_18-19 years',
            'demo_Age_20-24 years',
            'demo_Age_25-29 years',
            'demo_Age_30-34 years',
            'demo_Age_35-39 years'
        ],
        'Age_40-64': [
            'demo_Age_40-44 years',
            'demo_Age_45-49 years',
            'demo_Age_50-54 years',
            'demo_Age_55-59 years',
            'demo_Age_60-64 years'
        ],
        'Age_65+': [
            'demo_Age_65-69 years',
            'demo_Age_70- years'
        ],
        'High_SES': [
            'demo_Socioeconomy_Top executives',
            'demo_Socioeconomy_Self-employed',
        ],
        'Economic_Vulnerability': [
            'demo_Socioeconomy_Unemployed',
            'demo_Benefit type_Kontanthjaelp',
            'demo_Benefit type_Foertidspension'
        ],
            # Immigration
        'Immigrants': [
            'demo_Immigrants_Former Yugoslavia',
            'demo_Immigrants_Other Europe',
            'demo_Immigrants_Africa',
            'demo_Immigrants_Asia and Oceania',
            'demo_Immigrants_North America',
            'demo_Immigrants_New EU countries',
            'demo_Immigrants_South and Central America',
            'demo_Immigrants_Nordic',
            'demo_Immigrants_Not specified',
            'demo_Immigrants_Old EU countries',
            'demo_Immigrants_Turkey'
        ],
        # Citizenship
        'Citizenship_Nordic': [
            'demo_Citizenship_Nordic',
            'demo_Citizenship_Denmark'
        ],
        'Citizenship_Other_Europe': [
            'demo_Citizenship_Former Yugoslavia',
            'demo_Citizenship_Other Europe'
        ],
    }
    
    # Create merged features
    for new_feature, old_features in mappings.items():
        existing_cols = [col for col in old_features if col in df.columns]
        if len(existing_cols) >= 2:
            # Determine prefix based on feature type
            if 'Age' in new_feature:
                prefix = 'demo_'
            elif 'education' in new_feature.lower() or new_feature in ['Not_specified', 'Total_education']:
                prefix = 'demo_Education_'
            elif 'Benefit' in new_feature:
                prefix = 'demo_'
            else:
                prefix = 'demo_'
            
            df[f'{prefix}{new_feature}'] = df[existing_cols].sum(axis=1)
            print(f"Merged {new_feature}: {len(existing_cols)} columns")
    
    # Drop old columns
    cols_to_drop = [col for mapping in mappings.values() 
                    for col in mapping if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} original columns after merging")
    
    return df

# Aggregate demographics over years
def aggregate_demographics(demo_df):
    if demo_df.empty:
        return pd.DataFrame()
    
    id_cols = ['Gruppe', 'year']
    demo_cols = [col for col in demo_df.columns if col not in id_cols]
    
    agg_dict = {col: 'mean' for col in demo_cols}
    demo_agg = demo_df.groupby('Gruppe').agg(agg_dict).reset_index()
    
    rename_dict = {col: f'demo_{col}' for col in demo_cols}
    demo_agg = demo_agg.rename(columns=rename_dict)  
    demo_agg = merge_features(demo_agg)    
    
    return demo_agg

# Prepare features by merging election, temporal, and demographic data
def prepare_features(election_df, temporal_df, demo_df):   
    for df in [election_df, temporal_df]:
        if 'Gruppe' in df.columns:
            df['Gruppe'] = df['Gruppe'].astype(str)
    
    df = temporal_df.copy()
    
    # Only aggregate election features that aren't already in temporal_features
    election_cols = ['top1_share', 'top2_share', 'top2_margin', 'herfindahl_index', 'effective_num_parties']
    
    if 'Voter turnout' in election_df.columns:
        election_cols.append('Voter turnout')
    
    # Remove columns that already exist in temporal_df to avoid duplicates
    existing_temporal_cols = set(temporal_df.columns)
    election_cols = [col for col in election_cols if col not in existing_temporal_cols]
    
    if election_cols:
        agg_dict = {col: 'mean' for col in election_cols if col in election_df.columns}
        
        if agg_dict:  # Only aggregate if there are columns to aggregate
            election_agg = election_df.groupby('Gruppe').agg(agg_dict).reset_index()
            
            rename_dict = {col: f'elec_{col}' for col in election_cols if col in election_agg.columns}
            election_agg = election_agg.rename(columns=rename_dict)
            
            df = df.merge(election_agg, on='Gruppe', how='left')
            print(f"Merged {len(election_cols)} election features (avoiding duplicates)")
    else:
        print("Skipped election features (all already in temporal_features)")
    
    if not demo_df.empty:
        df = df.merge(demo_df, on='Gruppe', how='left')
        print(f"Merged demographic features for {len(df)} polling area groups")
    else:
        print("Warning: No demographic data to merge")
    
    return df

# Select features and target variable
def select_features(df, target_col='swing_state'):
    exclude_cols = [
        'Gruppe', 'ValgstedId', 'PollingAreaID', 'swing_state', 'swing_switches', 'swing_margin', 'swing_combined', 'switch_rate', 'party_switches', 'avg_top2_margin', 'max_top2_margin', 'min_top2_margin', 'std_top2_margin', 'elec_top1_share', 'elec_top2_share', 'elec_top2_margin', 'elec_herfindahl_index', 'elec_effective_num_parties', 'demo_Benefit type_Folkepension', 'demo_Benefit type_Modtager ikke ydelser', 'demo_Citizenship_Not specified', 'demo_Education_Not_specified', 'demo_Immigrants'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    y = df[target_col]
    
    return X, y, feature_cols


# Get top N features based on initial Random Forest importance
def get_top_features_initial(X, y, n_features=TOP_N_FEATURES, random_state=42):
    
    # Train a simple model to get feature importance
    rf_initial = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    
    rf_initial.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_initial.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = importance_df.head(n_features)['feature'].tolist()
    
    return top_features, importance_df

# Train Random Forest model
def train_random_forest(X, y, random_state=42):
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_train_pred = rf.predict(X_train)
    y_train_pred_proba = rf.predict_proba(X_train)[:, 1] 
    
    return rf, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, y_train_pred, y_train_pred_proba


# Calculate classification metrics
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


# Evaluate model performance and check for overfitting
def evaluate_model(rf, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, y_train_pred):
    # Calculate metrics for train and test sets
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Overfitting check
    train_test_diff = train_metrics['Accuracy'] - test_metrics['Accuracy']
    print(f"  Train-Test Gap: {train_test_diff:.4f}")
    if train_test_diff > 0.15:
        print(" Warning: Significant overfitting detected!")
    elif train_test_diff > 0.08:
        print(" Moderate overfitting detected")
    else:
        print(" Overfitting is minimal")

    print(classification_report(y_test, y_pred, target_names=['Non-Swing', 'Swing']))
    
    # Save metrics to CSV
    all_metrics = set(list(train_metrics.keys()) + list(test_metrics.keys()))

    metric_names = []
    train_values = []
    test_values = []

    for metric in sorted(all_metrics):
        metric_names.append(metric)
        train_values.append(train_metrics.get(metric, 0.0))
        test_values.append(test_metrics.get(metric, 0.0))

    metrics_df = pd.DataFrame({
        'Metric': metric_names,
        'Train': train_values,
        'Test': test_values
    })
    metrics_df.to_csv(OUTPUT_DIR / 'classification_metrics.csv', index=False)


def plot_difference_lollipop(comparison_df, top_n=20):
    top_features = comparison_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sort by absolute difference
    top_features = top_features.sort_values('Difference')
    
    # Colors based on positive/negative difference
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_features['Difference']]
    
    # Create lollipop chart
    ax.hlines(y=range(len(top_features)), xmin=0, xmax=top_features['Difference'], color='gray', alpha=0.4, linewidth=2)
    ax.scatter(top_features['Difference'], range(len(top_features)), color=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2, zorder=3)
    
    # Customize
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=10)
    ax.set_xlabel('Difference (Swing - Non-Swing)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Differences: Swing vs Non-Swing Areas', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='Higher in Swing'),
        Patch(facecolor='#3498db', edgecolor='black', label='Higher in Non-Swing')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_differences_lollipop.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'feature_differences_lollipop.pgf')
    plt.close()
    

# Analyze swing states with selected features
def analyze_swing_states(df, feature_cols):
    swing_areas = df[df['swing_state'] == 1]
    non_swing_areas = df[df['swing_state'] == 0]
    
    # Compare selected features
    comparison_data = []
    for col in feature_cols:
        if col in df.columns:
            swing_mean = swing_areas[col].mean()
            non_swing_mean = non_swing_areas[col].mean()
            comparison_data.append({
                'Feature': col,
                'Swing Mean': swing_mean,
                'Non-Swing Mean': non_swing_mean,
                'Difference': swing_mean - non_swing_mean,
                'Abs Difference': abs(swing_mean - non_swing_mean)
            })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('Abs Difference', ascending=False)
    comparison_df.to_csv(OUTPUT_DIR / 'swing_comparison_selected.csv', index=False)
    
    return comparison_df

# Save predictions to CSV
def save_predictions(df, rf, X):
    predictions_df = df[['Gruppe', 'swing_state']].copy()
    
    if 'ValgstedId' in df.columns:
        predictions_df['ValgstedId'] = df['ValgstedId']
    
    predictions_df['predicted_swing_state'] = rf.predict(X)
    predictions_df['swing_probability'] = rf.predict_proba(X)[:, 1]
    predictions_df['correct_prediction'] = (
        predictions_df['swing_state'] == predictions_df['predicted_swing_state']
    ).astype(int)
    
    predictions_df = predictions_df.sort_values('swing_probability', ascending=False)
    predictions_df.to_csv(OUTPUT_DIR / 'swing_predictions.csv', index=False)
    
    return predictions_df

# Analyze SHAP values for model interpretation
def analyze_shap_values(rf, X_train, X_test, feature_cols):
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf)
    
    # Calculate SHAP values for test set
    shap_values_test = explainer.shap_values(X_test)

    
    # For binary classification, we want class 1 (swing state)
    # Handle both list and 3D array formats
    if isinstance(shap_values_test, list):
        shap_values_test = shap_values_test[1]  # Select class 1
    elif len(shap_values_test.shape) == 3:
        shap_values_test = shap_values_test[:, :, 1]  # Select class 1 from 3D array
    
    
    # Summary plot (beeswarm) - Shows feature effects
    # Adjust figure dimensions - make it taller with less vertical spacing
    fig = plt.figure(figsize=(20, 12))  # wider, slightly less tall

    try:
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_values_test,
                base_values=np.zeros(len(X_test)),  # dummy base values
                data=X_test.values,
                feature_names=X_test.columns.tolist()
            ),
            show=False,
            plot_size=(20, 12),   # match figure size, makes it wider
            max_display=20,       # ensure all 20 features are shown
            s=60                  # maybe slightly smaller dots to avoid overlap
        )
    except:
        shap.summary_plot(
            shap_values_test,
            X_test,
            show=False,
            plot_size=(20, 12),   # width x height in inches
            max_display=20
        )
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    ax = plt.gca()
    cax = fig.axes[-1]
    cax.tick_params(labelsize=16)
    cax.yaxis.label.set_size(18)

    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)  
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'shap_summary_beeswarm.pgf', bbox_inches='tight')
    plt.close()
    print("SHAP beeswarm plot saved")
    
    # 3. Calculate mean absolute SHAP values for ranking
    # Now shap_values_test should be (samples, features)
    mean_abs_shap = np.abs(shap_values_test).mean(axis=0)

    
    # Ensure the shape matches
    assert mean_abs_shap.shape[0] == len(X_test.columns), \
        f"Shape mismatch: {mean_abs_shap.shape[0]} vs {len(X_test.columns)}"
    
    shap_importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    shap_importance_df.to_csv(OUTPUT_DIR / 'shap_feature_importance.csv', index=False)
    print("SHAP feature importance saved")
    
    return shap_importance_df, shap_values_test


# Run model for a specific swing method
def run_model_for_method(method, election_path, temporal_path, output_dir):
    # Load data for this method
    election_df = pd.read_csv(election_path)
    temporal_df = pd.read_csv(temporal_path)
    demo_df = load_demographics()
    
    # Aggregate demographics (same for all methods)
    if not demo_df.empty:
        demo_agg = aggregate_demographics(demo_df)
    else:
        demo_agg = pd.DataFrame()
    
    # Prepare features
    df = prepare_features(election_df, temporal_df, demo_agg)
    
    # Select features
    X_all, y, all_feature_cols = select_features(df)
    
    # Get top features
    top_features, initial_importance = get_top_features_initial(X_all, y, n_features=TOP_N_FEATURES)
    X = X_all[top_features].copy()
    
    # Train model
    rf, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, y_train_pred, y_train_pred_proba = train_random_forest(X, y)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Store key results
    results = {
        'method': method,
        'swing_count': y.sum(),
        'train_accuracy': train_metrics['Accuracy'],
        'test_accuracy': test_metrics['Accuracy'],
        'test_f1': test_metrics['F1-Score'],
        'test_roc_auc': test_metrics.get('ROC-AUC', 0),
        'train_test_gap': train_metrics['Accuracy'] - test_metrics['Accuracy'],
        'rf_model': rf,
        'X': X,
        'df': df,
        'top_features': top_features,
        'test_metrics': test_metrics
    }
    
    # Save method-specific results
    metrics_df = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Test': list(test_metrics.values())
    })
    metrics_df.to_csv(output_dir / f'{method}_metrics.csv', index=False)
    
    # Predictions
    predictions_df = save_predictions(df, rf, X)
    predictions_df.to_csv(output_dir / f'{method}_predictions.csv', index=False)
    
    return results

def main():
    swing_methods = ['switches', 'margin', 'combined']
    all_results = []

    for method in swing_methods:
        method_dir = Path("processed-data/features/swing_states") / method
        election_path = method_dir / "election_features_full.csv"
        temporal_path = method_dir / "temporal_features.csv"

        if not election_path.exists():
            print(f"Missing data for {method}")
            continue

        output_dir = OUTPUT_DIR / method
        output_dir.mkdir(exist_ok=True)

        result = run_model_for_method(
            method=method,
            election_path=election_path,
            temporal_path=temporal_path,
            output_dir=output_dir
        )

        all_results.append(result)

    # Compare methods
    comparison_df = pd.DataFrame([{
        'method': r['method'],
        'swing_states': r['swing_count'],
        'test_f1': r['test_f1'],
        'roc_auc': r['test_roc_auc'],
        'overfit_gap': r['train_test_gap']
    } for r in all_results])

    comparison_df.to_csv(OUTPUT_DIR / "method_comparison_summary.csv", index=False)

    # Select best
    best_row = comparison_df.loc[comparison_df['test_f1'].idxmax()]
    best_method = best_row['method']

    with open(OUTPUT_DIR / "best_method.txt", "w") as f:
        f.write(best_method)

    return comparison_df

if __name__ == "__main__":
    rf_model, results_df, feature_importance, predictions, shap_importance = main()