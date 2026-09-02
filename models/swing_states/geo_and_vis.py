import matplotlib as mpl
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import folium

mpl.use("pgf")
mpl.rcParams.update({
    "pgf.rcfonts": False,      # don't override LaTeX fonts
    "text.usetex": True,       # use LaTeX for all text
    "font.family": "serif",    # match LaTeX document
    "font.size": 16,           # readable size
})


YEARS = [2009, 2013, 2017, 2021]
ELECTION_DIR = Path("processed-data/elections")
SWING_DIR = Path("processed-data/features/swing_states/switches")
GEOJSON_PATH = Path("processed-data/geodata/afstemningsomraader2021_CPH_FRB.geojson")
OUTPUT_DIR = Path("models/swing_states/geo_and_vis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Major parties for analysis
MAJOR_PARTIES = ['A', 'V', 'C', 'F', 'Ø', 'O', 'B', 'I']
PARTY_NAMES = {
    'A': 'Socialdemokratiet', 'V': 'Venstre', 'C': 'Konservative',
    'F': 'SF', 'Ø': 'Enhedslisten', 'O': 'Dansk Folkeparti',
    'B': 'Radikale Venstre', 'I': 'Liberal Alliance'
}


def load_election_data():
    dfs = []
    for year in YEARS:
        file_path = ELECTION_DIR / f"relative_{year}.csv"
        df = pd.read_csv(file_path)
        df['year'] = year
        df['Gruppe'] = df['Gruppe'].astype(str)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def load_swing_features():
    temporal = pd.read_csv(SWING_DIR / "temporal_features.csv")
    temporal['Gruppe'] = temporal['Gruppe'].astype(str)

    return temporal

def load_geojson():
    with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    return geojson

def match_geojson_to_data(geojson, df):
    # Extract GeoJSON IDs
    geo_areas = []
    for feature in geojson['features']:
        props = feature['properties']
        # Handle kommune code: "0101" or "147"
        kommune = props['kommunekode']
        area_num = props['afstemningsomraadenummer']
        
        # Create Gruppe ID: remove leading zero from kommune, add area_num
        if kommune == "0101":
            gruppe_id = f"1010{area_num}"
        elif kommune == "0147":
            gruppe_id = f"147{area_num}"
        else:
            gruppe_id = f"{kommune}{area_num}"
        
        geo_areas.append({
            'gruppe_id': gruppe_id,
            'navn': props['navn'],
            'kommune': 'København' if kommune == "0101" else 'Frederiksberg'
        })
    
    geo_df = pd.DataFrame(geo_areas)
    
    # Merge with swing data
    merged = geo_df.merge(df, left_on='gruppe_id', right_on='Gruppe', how='left')
    
    return merged, geo_df

# Party Competition Analysis
def get_party_columns(df):
    party_letters = list('ABCDEFGHIJKLMNOPQRSTUVXYZØÅÆ123T')
    return [col for col in df.columns if col in party_letters]

def calculate_winner_and_runner_up(df):
    party_cols = get_party_columns(df)
    
    results = []
    for _, row in df.iterrows():
        votes = {col: row[col] for col in party_cols if pd.notna(row[col]) and row[col] > 0}
        if len(votes) < 2:
            continue
        
        sorted_parties = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        
        results.append({
            'Gruppe': row['Gruppe'],
            'year': row['year'],
            'winner': sorted_parties[0][0],
            'winner_share': sorted_parties[0][1],
            'second': sorted_parties[1][0],
            'second_share': sorted_parties[1][1],
            'margin': sorted_parties[0][1] - sorted_parties[1][1]
        })
    
    return pd.DataFrame(results)

def analyze_party_competition(election_df, swing_df):
    # Get competition data
    competition = calculate_winner_and_runner_up(election_df)
    competition = competition.merge(swing_df[['Gruppe', 'swing_state']], on='Gruppe')
    
    # Analysis 1: Which parties win in swing vs non-swing areas
    party_wins = competition.groupby(['swing_state', 'winner']).size().reset_index(name='wins')
    party_wins_pct = party_wins.groupby('swing_state').apply(
        lambda x: x.assign(win_pct=100 * x['wins'] / x['wins'].sum())
    ).reset_index(drop=True)
    
    # Analysis 2: Most common matchups in swing states
    swing_matchups = competition[competition['swing_state'] == 1].copy()
    swing_matchups['matchup'] = swing_matchups.apply(
        lambda x: '-'.join(sorted([x['winner'], x['second']])), axis=1
    )
    matchup_counts = swing_matchups['matchup'].value_counts().head(10)
    
    # Analysis 3: Party switches over time
    switch_patterns = []
    for gruppe in competition['Gruppe'].unique():
        area_data = competition[competition['Gruppe'] == gruppe].sort_values('year')
        if len(area_data) < 2:
            continue
        
        winners = area_data['winner'].tolist()
        swing = area_data['swing_state'].iloc[0]
        
        for i in range(len(winners) - 1):
            if winners[i] != winners[i+1]:
                switch_patterns.append({
                    'Gruppe': gruppe,
                    'from_party': winners[i],
                    'to_party': winners[i+1],
                    'year_from': area_data.iloc[i]['year'],
                    'year_to': area_data.iloc[i+1]['year'],
                    'swing_state': swing
                })
    
    switch_df = pd.DataFrame(switch_patterns)
    
    return {
        'party_wins': party_wins_pct,
        'matchup_counts': matchup_counts,
        'switch_patterns': switch_df,
        'competition_full': competition
    }


# Herfindahl Index and Effective Number of Parties Analysis
def calculate_herfindahl(df):
    party_cols = get_party_columns(df)
    party_shares = df[party_cols].fillna(0).values / 100
    return np.sum(party_shares ** 2, axis=1)

def analyze_herfindahl_patterns(election_df, swing_df):
    # Calculate Herfindahl for each year if not present
    if 'herfindahl_index' not in election_df.columns:
        election_df['herfindahl_index'] = calculate_herfindahl(election_df)
    
    # Merge with swing classifications
    merged = election_df.merge(swing_df[['Gruppe', 'swing_state']], on='Gruppe')
    
    # Average Herfindahl by swing status
    herf_summary = merged.groupby('swing_state')['herfindahl_index'].agg([
        'mean', 'median', 'std', 'min', 'max'
    ]).round(3)
    
    # Effective number of parties (1/H)
    merged['effective_parties'] = 1 / merged['herfindahl_index']
    
    eff_summary = merged.groupby('swing_state')['effective_parties'].agg([
        'mean', 'median', 'std'
    ]).round(2)
    
    return {
        'herfindahl_summary': herf_summary,
        'effective_parties_summary': eff_summary,
        'merged_data': merged
    }


# Geographic Clustering Analysis
def analyze_geographic_clustering(geo_df, swing_df):
    # Merge geographic and swing data
    merged = geo_df.merge(swing_df[['Gruppe', 'swing_state']], left_on='gruppe_id', right_on='Gruppe', how='left')
    
    # Summary by municipality
    by_kommune = merged.groupby('kommune').agg({
        'swing_state': ['sum', 'count', 'mean']
    }).round(3)
    by_kommune.columns = ['swing_count', 'total_areas', 'swing_proportion']
    
    # List swing state names
    swing_areas = merged[merged['swing_state'] == 1][['navn', 'kommune']].sort_values('kommune')
    
    return {
        'by_municipality': by_kommune,
        'swing_area_names': swing_areas,
        'merged_geo': merged
    }


# Party Competition Visualization
def plot_party_competition(competition_data):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.style.use('seaborn-v0_8-colorblind')
    
    # 1. Party wins by swing status
    ax = axes[0]
    wins_pivot = competition_data['party_wins'].pivot(
        index='winner', columns='swing_state', values='win_pct'
    ).fillna(0)
    wins_pivot = wins_pivot.loc[wins_pivot.sum(axis=1).nlargest(8).index]
    
    wins_pivot.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Party Win Rates: Swing vs Non-Swing States', fontsize=14, fontweight='bold')
    ax.set_xlabel('Party')
    ax.set_ylabel('Win Rate (%)')
    ax.legend(['Non-Swing', 'Swing'], title='Area Type')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Party switches over time
    ax = axes[1]
    switch_data = competition_data['switch_patterns']
    if len(switch_data) > 0:
        switch_summary = switch_data.groupby(['from_party', 'to_party']).size().nlargest(10)
        switch_summary.plot(kind='barh', ax=ax, color="#8fa7f1")
        ax.set_title('Most Common Party Switches (2009-2021)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Switches')
        ax.set_ylabel('From Party → To Party')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'party_competition_analysis.png', dpi=300, bbox_inches='tight')
    return fig


# Simple Map Visualization using Matplotlib
def create_simple_map_visualization(geojson, swing_df):

    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Map GeoJSON features to swing status
    swing_dict = dict(zip(swing_df['Gruppe'], swing_df['swing_state']))
    
    for feature in geojson['features']:
        props = feature['properties']
        kommune = props['kommunekode']
        area_num = props['afstemningsomraadenummer']
        
        if kommune == "0101":
            gruppe_id = f"1010{area_num}"
        elif kommune == "0147":
            gruppe_id = f"147{area_num}"
        else:
            gruppe_id = f"{kommune}{area_num}"
        
        # Get swing status
        is_swing = swing_dict.get(gruppe_id, 0)
        color = '#e74c3c' if is_swing else '#2ecc71'
        
        # Extract coordinates
        geom = feature['geometry']
        if geom['type'] == 'Polygon':
            coords = geom['coordinates'][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            ax.fill(lons, lats, color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
        elif geom['type'] == 'MultiPolygon':
            for polygon in geom['coordinates']:
                coords = polygon[0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                ax.fill(lons, lats, color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.6, label=f'Swing States (n={swing_df["swing_state"].sum()})'),
        Patch(facecolor='#2ecc71', alpha=0.6, label=f'Non-Swing States (n={(1-swing_df["swing_state"]).sum()})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    ax.set_title('Geographic Distribution of Swing States\nCopenhagen and Frederiksberg',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_axis_off()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'swing_states_map.png', dpi=300, bbox_inches='tight')
    return fig

# Folium Interactive Map
def create_folium_swing_map(geojson, swing_df, output_path=None):
    m = folium.Map(location=[55.6761, 12.5683], zoom_start=12, tiles='OpenStreetMap', height=800)
    
    swing_dict = dict(zip(swing_df['Gruppe'], swing_df['swing_state']))
    matched_swing = 0
    matched_non_swing = 0
    
    def get_color(swing_state):
        return '#e74c3c' if swing_state == 1 else '#2ecc71'
    
    def style_function(feature):
        swing_state = feature['properties']['swing_state']
        return {
            'fillColor': get_color(swing_state),
            'fillOpacity': 0.7,
            'color': 'black',
            'weight': 1.2
        }
    
    # Add properties
    for feature in geojson['features']:
        props = feature['properties']
        kommune = props['kommunekode']
        area_num = props['afstemningsomraadenummer']
        
        if kommune == "0101":
            gruppe_id = f"1010{area_num}"
        elif kommune == "0147":
            gruppe_id = f"147{area_num}"
        else:
            gruppe_id = f"{kommune}{area_num}"
        
        swing_status = swing_dict.get(gruppe_id, 0)
        feature['properties']['swing_state'] = swing_status
        feature['properties']['gruppe_id'] = gruppe_id
        feature['properties']['name'] = props.get('navn', props.get('name', 'Unknown'))
        
        if swing_status == 1:
            matched_swing += 1
        else:
            matched_non_swing += 1
    
    folium.GeoJson(
        geojson,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "gruppe_id", "swing_state"],
            aliases=["Name:", "Gruppe ID:", "Swing:"],
            localize=True,
            sticky=True,
            labels=True,
            style=(
                "background-color: white; "
                "color: black; "
                "padding: 10px;"
            )
        )
    ).add_to(m)
    
    # Legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 20px; left: 20px; width: 220px; 
                background-color: rgba(255,255,255,0.95); 
                border:2px solid #333; border-radius: 8px; 
                z-index:9999; font-size:13px; padding: 12px;">
    <b>Swing States Map</b><br>
    <span style="color:#e74c3c">█</span> Swing: <b>{matched_swing}</b><br>
    <span style="color:#2ecc71">█</span> Non-swing: <b>{matched_non_swing}</b>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Title
    title_html = '''
    <h3 align="center" style="font-size:24px; font-weight:bold; margin:20px 0">
    Geographic Distribution of Swing States<br>
    <small style="font-size:14px; color:#666">Copenhagen & Frederiksberg</small>
    </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    if output_path is None:
        output_path = Path.cwd() / 'models' / 'swing_states' / 'geo_and_vis' / 'swing_states_map.html'
    
    m.save(output_path)

    return m


def main():
    election_df = load_election_data()
    swing_df = load_swing_features()
    geojson = load_geojson()
    
    # Match geographic data
    matched_geo, geo_df = match_geojson_to_data(geojson, swing_df)
    
    # Analyze party competition
    competition_data = analyze_party_competition(election_df, swing_df)
    
    # Geographic clustering
    geo_data = analyze_geographic_clustering(geo_df, swing_df)
    
    # Create visualizations
    plot_party_competition(competition_data)
    create_simple_map_visualization(geojson, swing_df)
    # Replace your old call with:
    create_folium_swing_map(geojson, swing_df)

    # Party wins summary
    competition_data['party_wins'].to_csv(
        OUTPUT_DIR / 'party_wins_by_swing_status.csv', index=False
    )
    
    # Geographic summary
    geo_data['by_municipality'].to_csv(
        OUTPUT_DIR / 'swing_states_by_municipality.csv'
    )
    
    geo_data['swing_area_names'].to_csv(
        OUTPUT_DIR / 'swing_state_list.csv', index=False
    )

if __name__ == "__main__":
    main()