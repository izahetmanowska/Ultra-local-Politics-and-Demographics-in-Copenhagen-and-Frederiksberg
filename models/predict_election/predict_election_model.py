from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

EXCLUDE_COLS = [
    'Gruppe','year', 'winning_party','A_vote_share', 'C_vote_share', 'F_vote_share',
    'V_vote_share', 'Ø_vote_share', 'Municipality'
]

SELECTED_FEATURES = [
    'Education_Long_higher_education',
    'Education_Primary_lower_secondary',
    'Education_Vocational_training',
    'Housing tenure_People_Almen',
    'Housing tenure_People_Ejer',
    'Housing type_People_Apartment buildings',
    'People_on_benefits',
    'Seniors_65_plus',
    'Socioeconomy_Top executives',
    'Socioeconomy_Unemployed'
]
WINNING_PARTIES = ['A', 'V', 'C', 'F', 'Ø']

# Check to tell how unstable it is to use earlier election years demographic data to predict next election outcome

def validate_temporal_lag(df_combined, selected_features=SELECTED_FEATURES, parties=WINNING_PARTIES):
    """
    Test to see if using older demographics (4 years prior) can predict elections
    
    This validates using 2021 demographics to predict 2025 elections
    
    Validation tests:
    - 2009 demographics → 2013 election
    - 2013 demographics → 2017 election  
    - 2017 demographics → 2021 election
    
    Parameters:
    -----------
    df_combined : pd.DataFrame
        Combined dataframe with all years
    selected_features : list
        List of demographic features
    parties : list
        Parties to predict
    
    Returns:
    --------
    lag_results : dict
        Performance metrics for each lagged prediction
    """ 
    
    print("-" * 70)
    print("Temporal lag validation")
    print("Testing: Can demographics from year N predict election at N+4?")
    print("-" * 70)
    
    # Defining lagged pairs: (demographic year, election year)
    lag_pairs = [
        (2009, 2013),
        (2013, 2017),
        (2017, 2021)
    ]
    
    lag_results = {}
    temporal_rows = []
    summary_rows = []
    
    for demo_year, election_year in lag_pairs:
        print(f"\n{'-'*70}")
        print(f"TEST: {demo_year} demographics → {election_year} election")
        print('-'*70)
        
        # Get demographic data from earlier year
        df_demo = df_combined[df_combined['year'] == demo_year].copy()
        
        # Get election results from later year
        df_election = df_combined[df_combined['year'] == election_year].copy()
        
        # Merge on Gruppe (polling area)
        df_merged = df_demo[['Gruppe'] + selected_features].merge(
            df_election[['Gruppe'] + [f'{p}_vote_share' for p in parties]],
            on='Gruppe',
            how='inner'
        )
        
        print(f"Matched {len(df_merged)} polling areas")
        
        if len(df_merged) == 0:
            print("No matching polling areas found!")
            continue
        
        party_metrics = {}
        
        # training on same-year data from OTHER years, test on lagged pair
        # Train on all years exept the election year that is being tested
        train_years = [y for y in [2009, 2013, 2017, 2021] if y != election_year]
        df_train = df_combined[df_combined['year'].isin(train_years)].copy()
        
        for party in parties:
            target_col = f'{party}_vote_share'
            
            # Training data (same-year demographics → same-year election)
            X_train = df_train[selected_features].copy()
            y_train = df_train[target_col].copy()
            
            # Test data (old demographics → new election)
            X_test = df_merged[selected_features].copy()
            y_test = df_merged[target_col].copy()
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                cv=5,
                max_iter=20000,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred = np.clip(y_pred, 0, 100)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            party_metrics[party] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }

            temporal_rows.append({
                "period": f"{demo_year}→{election_year}",
                "party": party,
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            })
        
        lag_results[f"{demo_year}→{election_year}"] = party_metrics


        # Print summary
        print(f"\nPerformance Summary:")
        for party in parties:
            metrics = party_metrics[party]
            print(f"  Party {party}: MAE={metrics['mae']:.2f}%, R²={metrics['r2']:.3f}")
    
    # Overall summary
    print(f"\n{'-'*70}")
    print("Summary: Temporal Lag Performance")
    print('-'*70)
    
    for lag_pair, party_metrics in lag_results.items():
        avg_mae = np.mean([m['mae'] for m in party_metrics.values()])
        avg_r2 = np.mean([m['r2'] for m in party_metrics.values()])
        print(f"{lag_pair}: Avg MAE={avg_mae:.2f}%, Avg R²={avg_r2:.3f}")
    
    overall_mae = np.mean([m['mae'] for pm in lag_results.values() for m in pm.values()])
    overall_r2 = np.mean([m['r2'] for pm in lag_results.values() for m in pm.values()])
    
    print(f"\nOverall Average: MAE={overall_mae:.2f}%, R²={overall_r2:.3f}")
    print(f"\nInterpretation:")
    print(f"   This shows how well our 2021→2025 prediction might perform")
    print(f"   If MAE < 5%, predictions are quite reliable")
    print(f"   If MAE > 8%, consider this a rough estimate only")
    for lag_pair, party_metrics in lag_results.items():
        summary_rows.append({
        "period": lag_pair,
        "avg_mae": np.mean([m['mae'] for m in party_metrics.values()]),
        "avg_r2": np.mean([m['r2'] for m in party_metrics.values()])
        })

    df_temporal_metrics = pd.DataFrame(temporal_rows)
    df_temporal_summary = pd.DataFrame(summary_rows)

    #returns a dict of results
    return {
        "raw": lag_results,
        "metrics_df": df_temporal_metrics,
        "summary_df": df_temporal_summary
    }


#_________________________________________________________________________________________________
# Traditional Validation i.e. Same-Year Demographics
# Training on 2009-2017, Test on 2021 (using 2021 demographics)


def validate_same_year(df_combined, selected_features=SELECTED_FEATURES, parties=WINNING_PARTIES):
    """
    Traditional validation: same-year demographics → same-year election
    Train on 2009-2017, test on 2021
    
    This shows our model's performance under ideal conditions (correct demographics)
    """
    
    print("-" * 70)
    print("SAME-YEAR VALIDATION: Train 2009-2017 → Test 2021")
    print("(Using 2021 demographics → 2021 election)")
    print("-" * 70)

    # Split data
    df_train = df_combined[df_combined['year'].isin([2009, 2013, 2017])].copy()
    df_test = df_combined[df_combined['year'] == 2021].copy()
    
    print(f"\nTraining: {len(df_train)} samples from 2009, 2013, 2017")
    print(f"Testing: {len(df_test)} samples from 2021")
    
    results = {}
    all_predictions = pd.DataFrame()
    all_predictions['Gruppe'] = df_test['Gruppe'].values
    long_rows = []

    for party in parties:
        print(f"\n{'-'*70}")
        print(f"Party {party}")
        print('-'*70)
        
        target_col = f'{party}_vote_share'
        
        # Prepare data
        X_train = df_train[selected_features].copy()
        y_train = df_train[target_col].copy()
        X_test = df_test[selected_features].copy()
        y_test = df_test[target_col].copy()
        
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=5,
            max_iter=20000,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred = np.clip(y_pred, 0, 100)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        all_predictions[f'{party}_actual'] = y_test.values
        all_predictions[f'{party}_predicted'] = y_pred
        
        for a, p in zip(y_test.values, y_pred):
            long_rows.append({
                "party": party,
                "actual": a,
                "predicted": p
            })

        results[party] = {
            'model': model,
            'scaler': scaler,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"MAE:  {mae:.2f}%")
        print(f"RMSE: {rmse:.2f}%")
        print(f"R²:   {r2:.3f}")
    
    df_long = pd.DataFrame(long_rows)
    # Winner accuracy
    party_pred_cols = [f'{p}_predicted' for p in parties]
    all_predictions['predicted_winner'] = all_predictions[party_pred_cols].idxmax(axis=1).str.replace('_predicted', '')
    
    party_actual_cols = [f'{p}_actual' for p in parties]
    all_predictions['actual_winner'] = all_predictions[party_actual_cols].idxmax(axis=1).str.replace('_actual', '')
    
    accuracy = (all_predictions['predicted_winner'] == all_predictions['actual_winner']).mean()
    
    print(f"\n{'-'*70}")
    print(f"Winner Prediction Accuracy: {accuracy:.1%}")
    print('-'*70)
    
    return {
        'party_results': results,
        'predictions_wide': all_predictions,
        'predictions_long': df_long,
        'accuracy': accuracy
    }

# ________________________________________________________________________________________________
# Training Final Models on all years


def train_final_models(df_combined, selected_features=SELECTED_FEATURES, parties=WINNING_PARTIES):
    """
    Train final models on ALL years (2009-2021)
    These will be used with 2021 demographics to predict 2025 election
    """
    
    print("-" * 70)
    print("Final models: Training on 2009-2021")
    print("-" * 70)
    
    final_models = {}
    cv_rows = []
    
    for party in parties:
        print(f"\nTraining Party {party}...")
        
        target_col = f'{party}_vote_share'
        
        X = df_combined[selected_features].copy()
        y = df_combined[target_col].copy()
        
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=5,
            max_iter=20000,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_scaled, y)
        cv_r2 = model.score(X_scaled, y)

        final_models[party] = {
            'model': model,
            'scaler': scaler,
            'selected_features': selected_features,
            'cv_r2': cv_r2
        }

        cv_rows.append({
            "party": party,
            "cv_r2": cv_r2
        })
        
        print(f"  R² (CV): {model.score(X_scaled, y):.3f}")

    df_cv_r2 = pd.DataFrame(cv_rows)

    return final_models, df_cv_r2


# ________________________________________________________________________________
# Predicting 2025 using 2021 Demographics as Proxy

def predict_2025_with_proxy(df_combined_2021_dem_2025_elec, final_models, parties=WINNING_PARTIES, exclude_cols=EXCLUDE_COLS):
    """
    Predict 2025 election using 2021 demographics as proxy
    Data is assumed to be pre-combined into a single DataFrame

    Parameters:
    -----------
    df_2025_proxy : pd.DataFrame
        Combined dataframe with demographics from 2021 + 2025 election results
    final_models : dict
        Trained models from train_final_models()
    parties : list
        Parties to predict
    exclude_cols : list
        Columns to exclude from model features

    Returns:
    --------
    results_2025 : pd.DataFrame
        Predictions vs actual for 2025
    """

    print("-" * 70)
    print("2025 PREDICTION USING 2021 DEMOGRAPHICS AS PROXY")
    print("-" * 70)
    print("\nOBS: Using 2021 demographics to predict 2025 election")
    print("     Assumes demographics are stable over time")

    # Start from a copy to avoid side effects
    predictions_2025 = df_combined_2021_dem_2025_elec.copy()

    metrics_rows = []
    long_rows = []

    # Prediction loop
    for party in parties:
        print(f"\nPredicting Party {party}...")

        model_dict = final_models[party]
        model = model_dict["model"]
        scaler = model_dict["scaler"]
        selected_features = model_dict["selected_features"]

        # Ensure excluded columns are not used accidentally
        selected_features = [
            col for col in selected_features if col not in exclude_cols
        ]

        X = predictions_2025[selected_features].copy()


        # Scale & predict
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_pred = np.clip(y_pred, 0, 100)

        predictions_2025[f"{party}_predicted"] = y_pred

        # Metrics if actual results exist
        actual_col = f"{party}_vote_share"
        if actual_col in predictions_2025.columns:
            y_actual = predictions_2025[actual_col]

            mae = mean_absolute_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)

            metrics_rows.append({
                "party": party,
                "mae": mae,
                "r2": r2,
                "predicted_mean": y_pred.mean(),
                "actual_mean": y_actual.mean()
            })

            for a, p in zip(y_actual, y_pred):
                long_rows.append({
                    "party": party,
                    "actual": a,
                    "predicted": p
                })

            print(f"  MAE: {mae:.2f}%")
            print(f"  R²:  {r2:.3f}")
            print(f"  Predicted mean: {y_pred.mean():.1f}%")
            print(f"  Actual mean:    {y_actual.mean():.1f}%")


    df_2025_metrics = pd.DataFrame(metrics_rows)
    df_2025_long = pd.DataFrame(long_rows)

    # Winner prediction

    party_pred_cols = [f"{p}_predicted" for p in parties]
    predictions_2025["predicted_winner"] = (
        predictions_2025[party_pred_cols]
        .idxmax(axis=1)
        .str.replace("_predicted", "")
    )

    # Actual winners (if all available)
    party_actual_cols = [
        f"{p}_vote_share" for p in parties
        if f"{p}_vote_share" in predictions_2025.columns
    ]

    if len(party_actual_cols) == len(parties):
        predictions_2025["actual_winner"] = (
            predictions_2025[party_actual_cols]
            .idxmax(axis=1)
            .str.replace("_vote_share", "")
        )

        accuracy = (
            predictions_2025["predicted_winner"]
            == predictions_2025["actual_winner"]
        ).mean()

        print(f"\n{'-' * 70}")
        print(f"2025 WINNER PREDICTION ACCURACY: {accuracy:.1%}")
        print("-" * 70)

        confusion = pd.crosstab(
            predictions_2025["actual_winner"],
            predictions_2025["predicted_winner"],
            rownames=["Actual"],
            colnames=["Predicted"]
        )

        print("\nConfusion Matrix:")
        print(confusion)
    else:
        print("\nCannot calculate accuracy — missing some 2025 results")

    # Summary
    print(f"\n{'-' * 70}")
    print("PREDICTED 2025 RESULTS SUMMARY")
    print("-" * 70)

    print("\nPredicted Winners:")
    print(predictions_2025["predicted_winner"].value_counts())

    if "actual_winner" in predictions_2025.columns:
        print("\nActual Winners:")
        print(predictions_2025["actual_winner"].value_counts())

    return {
        "predictions_wide": predictions_2025,
        "predictions_long": df_2025_long,
        "metrics": df_2025_metrics,
        "confusion": confusion if 'actual_winner' in predictions_2025 else None,
        "accuracy": accuracy if 'actual_winner' in predictions_2025 else None
    }

#______________________________________________________________________________
# Plots 

def plot_temporal_mae(
    df_temporal,
    figsize=(10, 6),
    title="Temporal Lag Validation: Mean Absolute Error",
    save_path=None,
):
    """
    Plot MAE across temporal validation windows for each party,
    including the average MAE as a reference.

    Parameters
    ----------
    df_temporal : pd.DataFrame
        Columns: ['period', 'party', 'mae']
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str or None
        If provided, saves the figure (.png and .pgf).
    """

    # --- LaTeX / PGF Settings ---
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.rcfonts": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 13,
    })

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=df_temporal,
        x="period",
        y="mae",
        hue="party",
        marker="o",
        linewidth=1.8,
        ax=ax,
    )

    # Average MAE
    avg_mae = df_temporal.groupby("period")["mae"].mean()
    ax.plot(
        avg_mae.index,
        avg_mae.values,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="Average MAE",
    )

    ax.axhline(5, color="gray", linestyle=":", linewidth=1)

    ax.set_ylabel("MAE (percentage points)")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=18, pad=15)
    ax.legend(title="Party")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".pgf"))

    plt.show()

def plot_temporal_r2_heatmap(
    r2_matrix,
    figsize=(7, 4),
    cmap="RdYlGn",
    title="Temporal Lag Validation: $R^2$ Scores",
    save_path=None,
):
    """
    Heatmap of R² scores across temporal validation windows.

    Parameters
    ----------
    r2_matrix : pd.DataFrame
        Index: parties, Columns: temporal windows
    figsize : tuple
        Figure size.
    cmap : str
        Colormap.
    title : str
        Plot title.
    save_path : str or None
        Output path without extension.
    """

    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.rcfonts": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 13,
    })

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        r2_matrix,
        ax=ax,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("Election pair")
    ax.set_ylabel("Party")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".pgf"))

    plt.show()

def plot_predicted_vs_actual(
    df,
    party,
    figsize=(5, 5),
    title=None,
    save_path=None,
):
    """
    Scatter plot of predicted vs actual vote share for one party.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: ['actual', 'predicted']
    party : str
        Party label for the title.
    figsize : tuple
        Figure size.
    title : str or None
        Custom title.
    save_path : str or None
        Output path.
    """

    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.rcfonts": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        data=df,
        x="actual",
        y="predicted",
        alpha=0.7,
        ax=ax,
    )

    lims = [
        min(df["actual"].min(), df["predicted"].min()),
        max(df["actual"].max(), df["predicted"].max()),
    ]
    ax.plot(lims, lims, linestyle="--", color="gray")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Actual vote share (\\%)")
    ax.set_ylabel("Predicted vote share (\\%)")

    if title is None:
        title = f"{party}: Predicted vs Actual Vote Share"

    ax.set_title(title, fontsize=16, pad=10)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".pgf"))

    plt.show()

def plot_cv_r2(
    df_cv,
    figsize=(6, 4),
    title="Final Models: Cross-Validated $R^2$ (2009--2021)",
    save_path=None,
):
    """
    Bar chart of cross-validated R² for final models.

    Parameters
    ----------
    df_cv : pd.DataFrame
        Columns: ['party', 'cv_r2']
    """

    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.rcfonts": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 13,
    })

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=df_cv,
        x="party",
        y="cv_r2",
        color="gray",
        ax=ax,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("$R^2$")
    ax.set_xlabel("Party")
    ax.set_title(title, fontsize=16, pad=12)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".pgf"))

    plt.show()

def plot_confusion_matrix(
    confusion,
    figsize=(5, 4),
    cmap="Blues",
    title="2025 Election: Winner Prediction Confusion Matrix",
    save_path=None,
):
    """
    Plot confusion matrix for winner prediction.

    Parameters
    ----------
    confusion : pd.DataFrame
        Crosstab of actual vs predicted winners.
    """

    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.rcfonts": False,
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap=cmap,
        linewidths=0.4,
        ax=ax,
    )

    ax.set_xlabel("Predicted winner")
    ax.set_ylabel("Actual winner")
    ax.set_title(title, fontsize=16, pad=12)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".pgf"))

    plt.show()


