
#Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


EXCLUDE_COLS = ['Gruppe', 'winning_votes', 'winning_party_encoded','A_vote_share', 'C_vote_share', 'F_vote_share', 'V_vote_share', 'Ø_vote_share', 'winning_party' ]  # IDs and target variables

def get_df_info(df, name="DataFrame"):
    """
    A function that prints the relevant info about a DataFrame
    relevant info include: 
        - shape
        - first 5 rows
        - column rows
        - variable types
        - describtion, i.e. stats for the columns
        - the amount of missing values
    """
    print(f"___________Info for {name}___________")
    print(f"The shape of the DataFrame: {df.shape}\n")
    print(f"The top 5 rows of the df: \n{df.head()}\n")
    print(f"The columns in the df: \n{df.columns}\n")
    print(f"The variable types in the df: \n{df.dtypes}\n")
    print(f"The stats for the df:\n{df.describe()}\n")
    print(f"The missing values in the df:\n{df.isnull().sum()}")

def correlation_matrix(df, cols_to_exclude=EXCLUDE_COLS):
    # Selecting only numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Removing columns that shouldn't be in correlation analysis
    cols_to_exclude = cols_to_exclude
    correlation_cols = [col for col in numeric_cols if col not in cols_to_exclude]

    # Creating correlation matrix
    correlation_matrix = df[correlation_cols].corr()

    return correlation_matrix



def find_high_correlations(corr, threshold=0.8):
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = [
        (col, row, upper.loc[row, col])
        for col in upper.columns
        for row in upper.index
        if abs(upper.loc[row, col]) > threshold
    ]

    return pd.DataFrame(pairs, columns=['Feature 1', 'Feature 2', 'Correlation']) \
             .sort_values('Correlation', key=abs, ascending=False)



def plot_heatmap(
    corr,
    figsize=(18, 16),
    cmap="vlag",
    annot=True,
    fmt=".2f",
    title="Correlation Heatmap",
    save_path=None,
):
    """
    Generate a correlation heatmap with LaTeX/PGF support for publication-ready plots.

    Parameters
    ----------
    corr : correlation matrix 
        from the correlation_matrix function.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap used for the heatmap.
    annot : bool
        Whether to annotate each cell with the correlation value.
    fmt : str
        String formatting for annotations.
    title : str
        Title displayed on the heatmap.
    save_path : str or None
        If provided, saves the figure to this path (.pgf and .pdf).
    """

    # --- LaTeX / PGF Settings ---
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.rcfonts": False,      # don't override LaTeX fonts
        "text.usetex": True,       # use LaTeX for all text
        "font.family": "serif",    # match LaTeX document
        "font.size": 14,           # readable size
    })
    # Plot  
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        annot_kws={"size": 8},
        linewidths=0.3,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=26, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)


    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Save PNG with a NON-PGF backend ---
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")

        # --- Save PGF separately ---
        mpl.use("pgf")
        fig.savefig(save_path.with_suffix(".pgf"))
   

    plt.show()

# After correlation analysis drop list:
CORR_DROP_LIST =  ['Benefit type_Folkepension', 'Income_Gross income top20pct', 'Households_with_car']

def after_corr_drop(df):
    return df.drop(columns=CORR_DROP_LIST, errors="ignore")

import pandas as pd


#concating df's and aligning columns.

def align_columns_across_years(df_2009, df_2013, df_2017, df_2021, df_2025=None):
    """
    Ensure all dataframes have the same columns before concatenating
    Missing columns will be filled with NaN
    
    This is useful if different years have different parties or features
    """
    
    dfs = {
        2009: df_2009.copy(),
        2013: df_2013.copy(),
        2017: df_2017.copy(),
        2021: df_2021.copy()
    }
    
    if df_2025 is not None:
        dfs[2025] = df_2025.copy()
    
    # Get all unique columns across all years
    all_columns = set()
    for df in dfs.values():
        all_columns.update(df.columns)
    
    print(f"Total unique columns across all years: {len(all_columns)}")
    
    # Add missing columns to each dataframe (filled with 0)
    for year, df in dfs.items():
        missing_cols = all_columns - set(df.columns)
        if missing_cols:
            print(f"\nYear {year}: Adding {len(missing_cols)} missing columns")
            for col in missing_cols:
                df[col] = 0  # or use np.nan
        
        # Add year column
        df['year'] = year
        dfs[year] = df
    
    # Concatenate with aligned columns
    df_combined = pd.concat(dfs.values(), ignore_index=True)
    df_combined = df_combined.sort_values(['year', 'Gruppe']).reset_index(drop=True)
    
    print(f"\nFinal combined dataframe: {df_combined.shape}")
    print(f"Missing values by column (top 10):")
    missing_counts = df_combined.isnull().sum().sort_values(ascending=False)
    print(missing_counts[missing_counts > 0].head(10))
    
    return df_combined


# validation of combined df
def validate_combined_data(df_combined):
    """
    Run quick checks on the combined dataframe
    """

    print("VALIDATION CHECKS")
    print("=" * 60)
    
    # 1. Check for duplicate rows
    duplicates = df_combined.duplicated(subset=['Gruppe', 'year']).sum()
    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate Gruppe-Year combinations!")
    else:
        print("✓ No duplicate Gruppe-Year combinations")
    
    # 2. Check expected number of rows per year
    expected_rows = 58  # Your Copenhagen/Frederiksberg polling areas
    for year in df_combined['year'].unique():
        n_rows = len(df_combined[df_combined['year'] == year])
        if n_rows != expected_rows:
            print(f"Year {year}: {n_rows} rows (expected {expected_rows})")
        else:
            print(f"✓ Year {year}: {n_rows} rows")
    
    # 3. Check missing values by year
    print("\nMissing values by year:")
    for year in sorted(df_combined['year'].unique()):
        df_year = df_combined[df_combined['year'] == year]
        missing_pct = (df_year.isnull().sum().sum() / df_year.size) * 100
        print(f"  {year}: {missing_pct:.2f}% missing")
    
    # 4. Check data types
    print("\nColumn data types:")
    print(df_combined.dtypes.value_counts())
    
    return True




# before model check 

def check_preprocessing(df):
    """
    Check your data for issues before model
    """
    print("=" * 60)
    print("Preprocess validation")
    print("=" * 60)
    
    # Checking for missing values
    print("\n1. Missing Values:")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
        print("WARNING: Handle missing values before modeling!")
    else:
        print("No missing values")
    
    # Checking for infinite values
    print("\n2. Infinite Values:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print(inf_counts)
        print("WARNING: Remove infinite values!")
    else:
        print("No infinite values")
    
    # Checking feature scale/variance
    print("\n3. Feature Scales (sample of numeric columns):")
    sample_cols = numeric_cols[:5]  # Check first 5 numeric columns
    print(df[sample_cols].describe().loc[['mean', 'std', 'min', 'max']])
    print("\nIf scales vary widely (e.g., 0-1 vs 0-10000), standardize features")
    
    # 4. Check for constant/near-constant features
    print("\n4. Low Variance Features:")
    low_var = []
    for col in numeric_cols:
        if df[col].std() < 0.01:
            low_var.append(col)
    
    if low_var:
        print(f"These features have very low variance: {low_var}")
    else:
        print("All features have reasonable variance!")
    
    # 5. Check target variable distribution (for classification)<-REMOVE IF DOING REGRESSION
    if 'winning_party_encoded' in df.columns:
        print("\n5. Target Variable Distribution:")
        print(df['winning_party_encoded'].value_counts().sort_index())
        print("\nIf severe imbalance is detected then class_weight='balanced' will help")
    
    # 6. Check for duplicate rows
    print("\n6. Duplicate Rows:")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate rows found!")
    else:
        print("No duplicate rows")
    
    print("\n" + "-" * 60)
    return True




# REGRESSION - Feature Selection Predicting Vote Shares

def feature_selection_regression(df, major_parties=['A', 'V', 'C', 'F', 'Ø'], 
                                 years=[2009, 2013, 2017, 2021]):
    """
    Feature selection using ElasticNet regression
    Predicts vote share for each major party separately
    """
    print("\n" + "-" * 60)
    print("PATH 2: REGRESSION FEATURE SELECTION")
    print("-" * 60)
    
    # Check if vote share columns exist
    vote_cols = [f'{party}_vote_share' for party in major_parties]
    missing_cols = [col for col in vote_cols if col not in df.columns]
    if missing_cols:
        print(f" WARNING: Missing vote share columns: {missing_cols}")
        print("You need columns like 'A_vote_share', 'V_vote_share', etc.")
        return None
    
    # Prepare features
    exclude_cols = ['Gruppe', 'Municipality', 'winning_party', 'winning_votes', 
                    'winning_party_encoded', 'year'] + vote_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    results_by_party = {}
    all_features_importance = {}
    
    for party in major_parties:
        print(f"\n{'='*60}")
        print(f"PARTY: {party}")
        print('='*60)
        
        target_col = f'{party}_vote_share'
        results_by_year = {}
        
        # Option A: Train separate model for each year
        for year in years:
            print(f"\n--- Year: {year} ---")
            df_year = df[df['year'] == year].copy()
            
            X = df_year[feature_cols].copy()
            y = df_year[target_col]
            
            # Handle missing values
            for col in X.columns:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
            
            # Train ElasticNet
            model = ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                cv=5,
                max_iter=10000,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_scaled, y)
            
            # Get feature importance (absolute coefficients)
            importance = np.abs(model.coef_)
            
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            results_by_year[year] = {
                'model': model,
                'importance': feature_importance,
                'features_selected': feature_importance[feature_importance['importance'] > 0.01]['feature'].tolist(),
                'r2': model.score(X_scaled, y)
            }
            
            # Track across years and parties
            for idx, row in feature_importance.iterrows():
                feat = row['feature']
                key = f"{party}_{feat}"
                if key not in all_features_importance:
                    all_features_importance[key] = []
                all_features_importance[key].append(row['importance'])
            
            print(f"R² score: {results_by_year[year]['r2']:.3f}")
            print(f"Features selected: {len(results_by_year[year]['features_selected'])}")
            print(f"Top 5 features:")
            print(feature_importance.head())
        
        # Option B: Train on all years pooled
        print(f"\n--- All Years Pooled (2009-2021) ---")
        X_all = df[feature_cols].copy()
        y_all = df[target_col]
        
        for col in X_all.columns:
            if X_all[col].isna().any():
                X_all[col] = X_all[col].fillna(X_all[col].median())
        
        scaler_all = StandardScaler()
        X_all_scaled = scaler_all.fit_transform(X_all)
        X_all_scaled = pd.DataFrame(X_all_scaled, columns=feature_cols, index=X_all.index)
        
        model_all = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=5,
            max_iter=10000,
            random_state=42,
            n_jobs=-1
        )
        
        model_all.fit(X_all_scaled, y_all)
        
        importance_all = np.abs(model_all.coef_)
        feature_importance_all = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_all
        }).sort_values('importance', ascending=False)
        
        print(f"\nR² score: {model_all.score(X_all_scaled, y_all):.3f}")
        print(f"Top 10 features:")
        print(feature_importance_all.head(10))
        
        results_by_party[party] = {
            'by_year': results_by_year,
            'pooled': {'model': model_all, 'importance': feature_importance_all, 
                      'scaler': scaler_all}
        }
    
    # Aggregate: Which features are important across multiple parties?
    feature_party_count = {}
    for key in all_features_importance.keys():
        party, feat = key.split('_', 1)
        if feat not in feature_party_count:
            feature_party_count[feat] = {'parties': [], 'mean_importance': []}
        feature_party_count[feat]['parties'].append(party)
        feature_party_count[feat]['mean_importance'].append(np.mean(all_features_importance[key]))
    
    cross_party_importance = []
    for feat, data in feature_party_count.items():
        cross_party_importance.append({
            'feature': feat,
            'n_parties': len(data['parties']),
            'parties': ', '.join(data['parties']),
            'mean_importance': np.mean(data['mean_importance'])
        })
    
    cross_party_df = pd.DataFrame(cross_party_importance).sort_values(
        ['n_parties', 'mean_importance'], ascending=[False, False]
    )
    
    print(f"\nFeatures Important Across Multiple Parties:")
    print(cross_party_df.head(15))
    
    return {
        'by_party': results_by_party,
        'cross_party': cross_party_df,
        'feature_cols': feature_cols
    }



# Comparing top feature selection

def select_top_features(regression_results, n_features=10):
    """
    Given results from recommend top N features
    """
    print("\n" + "-" * 60)
    print(f"RECOMMENDED TOP {n_features} FEATURES")
    print("-" * 60)
    
    print("\nFrom Regression (Cross-Party Importance):")
    top_reg = regression_results['cross_party'].head(n_features)
    print(top_reg)
    
    # Combine recommendations
    selected_features = set()
    
    if regression_results:
        selected_features.update(
            regression_results['cross_party'].head(n_features)['feature'].tolist()
        )
    
    print(f"\nFINAL RECOMMENDED FEATURES ({len(selected_features)} total):")
    for i, feat in enumerate(sorted(selected_features), 1):
        print(f"{i}. {feat}")
    
    


    return list(selected_features)



