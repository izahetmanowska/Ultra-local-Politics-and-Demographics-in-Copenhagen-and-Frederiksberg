import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
import ptitprince as pt
import numpy as np
import os

# Publication-quality serif style
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.rcfonts": False,      # don't override LaTeX fonts
    "text.usetex": True,       # use LaTeX for all text
    "font.family": "serif",    # match LaTeX document
    "font.size": 20,           # readable size
})
sns.set_palette("husl")

# Electoral features
electoral = pd.read_csv('processed-data/features/swing_states/switches/election_features_full.csv')

# Demographics for all years - first column is Gruppe
demo_2009 = pd.read_csv('processed-data/demographics/relative_2009.csv')
demo_2013 = pd.read_csv('processed-data/demographics/relative_2013.csv')
demo_2017 = pd.read_csv('processed-data/demographics/relative_2017.csv')
demo_2021 = pd.read_csv('processed-data/demographics/relative_2021.csv')

# Rename Gruppe to PollingAreaID for consistency
for df in [demo_2009, demo_2013, demo_2017, demo_2021]:
    df.rename(columns={'Gruppe': 'PollingAreaID'}, inplace=True)
    df['PollingAreaID'] = df['PollingAreaID'].astype(str)

# Clustering data
clusters = pd.read_csv('processed-data/characterising-neighbourhoods/05_clusters.csv', dtype={'PollingAreaID': str})

print(f"Demo 2021 shape: {demo_2021.shape}")
print(f"Demo 2021 columns (first 10): {demo_2021.columns[:10].tolist()}")

# =============================================================================
# CREATE AGGREGATED FEATURES (matching your model)
# =============================================================================

def create_aggregated_features(df):
    """Create aggregated features matching the model"""
    df = df.copy()
    
    # Education
    df['Long_higher_education'] = (
        df['Education_18-29 years_Long higher education'] +
        df['Education_30-70+ years_Long higher education']
    )
    
    df['Vocational_training'] = (
        df['Education_18-29 years_Vocational training'] +
        df['Education_30-70+ years_Vocational training']
    )
    
    df['Primary_and_lower_secondary'] = (
        df['Education_18-29 years_Primary and lower secondary'] +
        df['Education_30-70+ years_Primary and lower secondary']
    )
    
    # Age groups
    df['Age_0-17'] = (
        df['Age_0-4 years'] +
        df['Age_5-9 years'] +
        df['Age_10-14 years'] +
        df['Age_15-17 years']
    )
    
    df['Age_18-39'] = (
        df['Age_18-19 years'] +
        df['Age_20-24 years'] +
        df['Age_25-29 years'] +
        df['Age_30-34 years'] +
        df['Age_35-39 years']
    )
    
    df['Age_40-64'] = (
        df['Age_40-44 years'] +
        df['Age_45-49 years'] +
        df['Age_50-54 years'] +
        df['Age_55-59 years'] +
        df['Age_60-64 years']
    )
    
    df['Age_65+'] = (
        df['Age_65-69 years'] +
        df['Age_70- years']
    )
    
    # Socioeconomic
    df['High_SES'] = (
        df['Socioeconomy_Top executives'] +
        df['Socioeconomy_Self-employed']
    )
    
    df['Economic_Vulnerability'] = (
        df['Socioeconomy_Unemployed'] +
        df['Benefit type_Kontanthjaelp'] +
        df['Benefit type_Foertidspension']
    )
    
    return df

# Apply to all years
demo_2009 = create_aggregated_features(demo_2009)
demo_2013 = create_aggregated_features(demo_2013)
demo_2017 = create_aggregated_features(demo_2017)
demo_2021 = create_aggregated_features(demo_2021)

# Add year column
demo_2009['year'] = 2009
demo_2013['year'] = 2013
demo_2017['year'] = 2017
demo_2021['year'] = 2021

# Combine all years
demo_all = pd.concat([demo_2009, demo_2013, demo_2017, demo_2021], ignore_index=True)

print(f"Combined demographics shape: {demo_all.shape}")
print(f"Years in demo_all: {demo_all['year'].unique()}")

# =============================================================================
# PREPARE DATA FOR VISUALIZATIONS
# =============================================================================

# Get electoral data for each year
electoral_by_year = []
for year in [2009, 2013, 2017, 2021]:
    year_data = electoral[electoral['year'] == year].copy()
    electoral_by_year.append(year_data)

electoral_all = pd.concat(electoral_by_year, ignore_index=True)

# Ensure PollingAreaID is string
if 'PollingAreaID' not in electoral_all.columns:
    electoral_all['PollingAreaID'] = electoral_all['Gruppe'].astype(str)
else:
    electoral_all['PollingAreaID'] = electoral_all['PollingAreaID'].astype(str)

# Merge with demographics
merged_all = electoral_all.merge(demo_all, on=['PollingAreaID', 'year'], how='left')

print(f"Merged data shape: {merged_all.shape}")

# Identify Frederiksberg (Gruppe starts with 147)
merged_all['is_frederiksberg'] = merged_all['PollingAreaID'].str.startswith('147')
merged_all['Municipality'] = merged_all['is_frederiksberg'].map({True: 'Frederiksberg', False: 'Copenhagen'})

# Split data
cph = merged_all[~merged_all['is_frederiksberg']]
frb = merged_all[merged_all['is_frederiksberg']]

print(f"\nCopenhagen areas: {cph['PollingAreaID'].nunique()}")
print(f"Frederiksberg areas: {frb['PollingAreaID'].nunique()}")

# =============================================================================
# VISUALIZATION 1: RAINCLOUD PLOTS FOR CPH VS FRB
# =============================================================================

# Select features for raincloud plots
features = {
    'Young Adults (18-39%)': 'Age_18-39',
    'Elderly (65+%)': 'Age_65+',
    'Andel Housing (%)': 'Housing tenure_People_Andel',
    'Long Higher Education (%)': 'Long_higher_education',
    'Voter Turnout (%)': 'Voter turnout',
    'Economic Vulnerability (%)': 'Economic_Vulnerability'
}

# Create raincloud plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
axes = axes.flatten()

cph_color = "#2580B7"  # Brighter, more saturated blue (was #021671)
frb_color = "#9EBE5B"  # Brighter, more saturated green (was #036318)

for idx, (feature_name, column) in enumerate(features.items()):
    ax = axes[idx]
    
    # Prepare data for raincloud plot
    plot_data = merged_all[['Municipality', column]].dropna()
    # Sort by Municipality to ensure Copenhagen is first, Frederiksberg second
    plot_data = plot_data.sort_values('Municipality')
    
    # Create raincloud plot using ptitprince WITHOUT points but WITH box plot
    pt.RainCloud(
        x='Municipality', 
        y=column, 
        data=plot_data,
        palette=[cph_color, frb_color],  # Use list instead of dictionary
        bw=0.2,  # bandwidth for kernel density
        width_viol=0.6,
        orient='h',
        ax=ax,
        alpha=0.8,
        dodge=True,
        point_size=0,  # Set to 0 to remove dots
        rain_alpha=0,   # Make rain (dots) transparent
        linewidth=0.5,
        box_showfliers=False  # Don't show outliers
    )
    
    ax.set_xlabel(feature_name, fontsize=16)  # Increased from 14
    ax.set_ylabel('')
    ax.set_yticklabels([])  # Remove y-axis labels
    ax.grid(axis='x', alpha=0.3)  # More visible grid
    ax.tick_params(axis='both', which='major', labelsize=11)  # Smaller tick labels
    
    # Remove all automatic text annotations from ptitprince
    for text in ax.texts:
        text.set_visible(False)
    
    # Expand y-axis limits to prevent cutoff
    ax.set_ylim(-0.6, 1.6)

# Create custom legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=cph_color, edgecolor='black', label='Copenhagen', alpha=0.7),
    Patch(facecolor=frb_color, edgecolor='black', label='Frederiksberg', alpha=0.7)
]

# Add legend to the figure
fig.legend(handles=legend_elements, loc='upper right', fontsize=13, 
           frameon=True, shadow=True, bbox_to_anchor=(1.1, 0.9))

fig.suptitle('Copenhagen vs Frederiksberg: Distribution Comparison Across Demographics', 
             fontsize=37, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=1, w_pad=2)
plt.savefig('cph_vs_frb_raincloud.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('cph_vs_frb_raincloud.pgf', bbox_inches='tight', facecolor='white')

print("✓ Raincloud plot complete: cph_vs_frb_raincloud.png")

# =============================================================================
# VISUALIZATION 2: MOSAIC PLOT - SWING STATES ACROSS CLUSTERS (EQUAL WIDTHS)
# =============================================================================

# Get unique swing state status per area
swing_status = electoral_all.groupby('PollingAreaID')['swing_state'].first().reset_index()

# Merge clusters with swing states
clusters['PollingAreaID'] = clusters['PollingAreaID'].astype(str)
cluster_swing = clusters.merge(swing_status, on='PollingAreaID', how='inner')

print(f"\nCluster-swing merge: {len(cluster_swing)} areas")

# Create labels for mosaic with cluster numbers
cluster_swing['Cluster_Label'] = cluster_swing['Cluster_kMeans'].map({
    0: '0 (Young Urban)',
    1: '1 (Social Housing)',
    2: '2 (Suburban)',
    3: '3 (New International)',
    4: '4 (Educated Elite)'
})

cluster_swing['Swing_Label'] = cluster_swing['swing_state'].map({
    0: 'Non-Swing',
    1: 'Swing'
})

# Define colors for each cluster
COLOURS = [
    "#2580B7",  # Blue (cluster 0 - Young Urban)
    "#179E86",  # Turquoise (1 - Social Housing)
    "#9EBE5B",  # Light Green (2 - Suburban)
    "#F59B11",  # Yellow/Orange (3 - New International)
    "#F24D00",  # Orange (4 - Educated Elite)
]

# Create mosaic plot with equal widths
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

# Calculate proportions within each cluster
cluster_props = cluster_swing.groupby(['Cluster_Label', 'Swing_Label']).size().unstack(fill_value=0)
cluster_props = cluster_props.div(cluster_props.sum(axis=1), axis=0)

# Sort by cluster number to ensure 0-4 order
cluster_order = ['0 (Young Urban)', '1 (Social Housing)', '2 (Suburban)', '3 (New International)', '4 (Educated Elite)']
cluster_props = cluster_props.reindex(cluster_order)

# Create stacked bar chart to simulate mosaic with equal widths
x_pos = np.arange(len(cluster_props))
width = 0.8

# Plot non-swing first (bottom)
bars_non_swing = ax.bar(x_pos, cluster_props['Non-Swing'], width, 
                        label='Non-Swing', color="#059424", alpha=0.85, 
                        edgecolor='black', linewidth=0.5)

# Plot swing on top
bars_swing = ax.bar(x_pos, cluster_props['Swing'], width, 
                    bottom=cluster_props['Non-Swing'],
                    label='Swing', color="#BB0505", alpha=0.85, 
                    edgecolor='black', linewidth=0.5)

# Formatting
ax.set_ylabel('Proportion', fontsize=18)
ax.set_xlabel('Cluster', fontsize=18, labelpad=10)
ax.set_title('Swing States Appear Across All Demographic Clusters', 
             fontsize=29, pad=35, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(cluster_props.index, fontsize=13, rotation=15)
ax.tick_params(axis='y', which='major', labelsize=13)

# Color each x-tick label
for i, (tick_label, label_text) in enumerate(zip(ax.get_xticklabels(), cluster_props.index)):
    tick_label.set_color(COLOURS[i])  # Use i directly since we sorted by cluster order

ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Add percentage labels
for i, cluster in enumerate(cluster_props.index):
    # Non-swing label
    non_swing_pct = cluster_props.loc[cluster, 'Non-Swing']
    if non_swing_pct > 0.05:  # Only show if > 5%
        ax.text(i, non_swing_pct/2, f'{non_swing_pct:.1%}', 
                ha='center', va='center', fontsize=11, color='white')
    
    # Swing label
    swing_pct = cluster_props.loc[cluster, 'Swing']
    if swing_pct > 0.05:  # Only show if > 5%
        ax.text(i, non_swing_pct + swing_pct/2, f'{swing_pct:.1%}', 
                ha='center', va='center', fontsize=11, color='white')

# Enhanced legend - positioned in upper right, outside the plot area
legend = ax.legend(fontsize=12, frameon=True, shadow=True, 
                   loc='upper right', bbox_to_anchor=(1.23, 1))
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('swing_clusters_mosaic.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('swing_clusters_mosaic.pgf', bbox_inches='tight', facecolor='white')

print("✓ Mosaic plot complete: swing_clusters_mosaic.png")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nCopenhagen vs Frederiksberg Means:")
for feature_name, column in features.items():
    cph_mean = merged_all[~merged_all['is_frederiksberg']][column].mean()
    frb_mean = merged_all[merged_all['is_frederiksberg']][column].mean()
    diff = frb_mean - cph_mean
    print(f"{feature_name:30s}: CPH={cph_mean:6.2f}  FRB={frb_mean:6.2f}  Diff={diff:+6.2f}")

print("\nSwing State Distribution by Cluster:")
cluster_counts = cluster_swing.groupby(['Cluster_Label', 'Swing_Label']).size().unstack(fill_value=0)
cluster_pct = (cluster_counts['Swing'] / cluster_counts.sum(axis=1) * 100)
for cluster in cluster_counts.index:
    swing_count = cluster_counts.loc[cluster, 'Swing']
    total = cluster_counts.loc[cluster].sum()
    pct = cluster_pct[cluster]
    print(f"{cluster:20s}: {swing_count:3d}/{total:3d} ({pct:5.1f}% swing)")

print("\n" + "="*60)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*60)
print(f"\nFiles created:")
print("1. cph_vs_frb_raincloud.png")
print("2. swing_clusters_mosaic.png")