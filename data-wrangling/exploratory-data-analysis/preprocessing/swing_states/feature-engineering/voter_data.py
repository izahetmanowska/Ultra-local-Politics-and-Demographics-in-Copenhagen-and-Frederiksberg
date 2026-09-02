import pandas as pd
import numpy as np
from pathlib import Path

YEARS = [2009, 2013, 2017, 2021]
DATA_DIR = Path("processed-data/elections")
OUTPUT_DIR = Path("processed-data/features/swing_states")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(years=YEARS):
    dfs = []
    for year in years:
        file_path = DATA_DIR / f"relative_{year}.csv"
        df = pd.read_csv(file_path, sep=',')
        df['year'] = year
        df['Gruppe'] = df['Gruppe'].astype(str)  # Ensure consistent ID type
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    return combined_df

def get_party_columns(df):
    party_letters = list('ABCDEFGHIJKLMNOPQRSTUVXYZØÅÆ123T')  # Fixed: T included
    party_cols = [col for col in df.columns if col in party_letters]

    return party_cols

def validate_party_shares(df, party_cols):
    party_shares = df[party_cols].fillna(0)
    row_sums = party_shares.sum(axis=1)

    # Normalize if sums deviate significantly from 100%
    if row_sums.min() < 95 or row_sums.max() > 105:
        df[party_cols] = party_shares.div(row_sums, axis=0) * 100
    
    return df

def calculate_top2(df):
    party_cols = get_party_columns(df)
    party_shares = df[party_cols].fillna(0).values
    sorted_shares = np.sort(party_shares, axis=1)[:, ::-1]
    df['top1_share'] = sorted_shares[:, 0]
    df['top2_share'] = sorted_shares[:, 1]
    df['top2_margin'] = df['top1_share'] - df['top2_share']

    return df

def calculate_herfindahl_index(df):
    party_cols = get_party_columns(df)
    party_shares = df[party_cols].fillna(0).values / 100  # Convert % to proportions
    df['herfindahl_index'] = np.sum(party_shares ** 2, axis=1)
    return df

def calculate_effective_parties(df):
    if 'herfindahl_index' not in df.columns:
        df = calculate_herfindahl_index(df)
    df['effective_num_parties'] = 1 / df['herfindahl_index']

    return df

def get_dominant_party(df):
    party_cols = get_party_columns(df)
    df['dominant_party'] = df[party_cols].idxmax(axis=1)

    return df

def calculate_party_switches(df):
    if 'dominant_party' not in df.columns:
        df = get_dominant_party(df)
    
    switch_list = []
    for gruppe_id in df['Gruppe'].unique():
        gruppe_data = df[df['Gruppe'] == gruppe_id].sort_values('year')
        if len(gruppe_data) < 2:
            continue
        parties = gruppe_data['dominant_party'].values
        switches = np.sum(parties[:-1] != parties[1:])
        switch_list.append({
            'Gruppe': gruppe_id,
            'num_elections': len(parties),
            'party_switches': switches,
            'switch_rate': switches / (len(parties) - 1)
        })
    switch_df = pd.DataFrame(switch_list)

    return switch_df

def calculate_avg_top2_margin(df):
    if 'top2_margin' not in df.columns:
        df = calculate_top2(df)
    margin_stats = df.groupby('Gruppe')['top2_margin'].agg([
        ('avg_top2_margin', 'mean'),
        ('min_top2_margin', 'min'),
        ('max_top2_margin', 'max'),
        ('std_top2_margin', 'std')
    ]).reset_index()

    return margin_stats

def define_swing_states(temporal_features, method='combined', threshold=None):
    if threshold is None:
        threshold = {'switches_min': 1, 'margin_max': 0.15}
    
    df = temporal_features.copy()
    
    # Switch-based
    df['swing_switches'] = (df['party_switches'] >= threshold['switches_min']).astype(int)
    
    margin_prop = df['avg_top2_margin'] / 100.0
    df['swing_margin'] = (margin_prop < threshold['margin_max']).astype(int)
    
    # Combined
    df['swing_combined'] = (
        (df['swing_switches'] == 1) & (df['swing_margin'] == 1)
    ).astype(int)

    swing_map = {
        'switches': 'swing_switches',
        'margin': 'swing_margin',
        'combined': 'swing_combined'
    }
    df['swing_state'] = df[swing_map[method]]
    return df

def main():  
    # Load data
    df = load_data()
    
    party_cols = get_party_columns(df)
    
    # Per-election features (already relative shares)
    df = validate_party_shares(df, party_cols)
    df = (df.pipe(calculate_top2)
          .pipe(calculate_herfindahl_index)
          .pipe(calculate_effective_parties)
          .pipe(get_dominant_party))
    
    # Temporal features (across elections per polling area)
    switch_df = calculate_party_switches(df)
    margin_df = calculate_avg_top2_margin(df)
    temporal_features = switch_df.merge(margin_df, on='Gruppe', how='outer')
    
    # Test ALL 3 swing state definitions
    swing_methods = ['switches', 'margin', 'combined']
    thresholds = {'switches_min': 1, 'margin_max': 0.10}
    
    all_results = {}
    
    for method in swing_methods:
        print(f"\n🔄 Processing swing method: {method}")
        
        # Define swing states for this method
        temp_df = define_swing_states(
            temporal_features.copy(),
            method=method,
            threshold=thresholds
        )
        
        swing_count = temp_df['swing_state'].sum()
        print(f"  ✓ {method}: {swing_count} swing states identified")
        
        # Merge back to full dataset
        df_full = df.merge(temp_df, on='Gruppe', how='left')
        
        # Save method-specific files
        method_dir = OUTPUT_DIR / method
        method_dir.mkdir(exist_ok=True)
        
        temp_df.to_csv(method_dir / 'temporal_features.csv', index=False)
        df_full.to_csv(method_dir / 'election_features_full.csv', index=False)
        
        all_results[method] = {
            'temporal_features': temp_df,
            'full_features': df_full,
            'swing_count': swing_count
        }
    
    # Summary across all methods
    summary_data = []
    for method, results in all_results.items():
        summary_data.append({
            'method': method,
            'swing_states_count': results['swing_count'],
            'swing_states_pct': f"{results['swing_count']/len(results['temporal_features'])*100:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / 'swing_methods_comparison.csv', index=False)
    print(f"\nSUMMARY: {summary_df}")
    
    return all_results

if __name__ == "__main__":
    results = main()