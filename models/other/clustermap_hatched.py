import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE (unchanged)
# =============================================================================
mpl.rcParams.update({
    "pgf.rcfonts": False,
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
})
sns.set_palette("husl")

COLOURS = ["#2580B7", "#179E86", "#9EBE5B", "#F59B11", "#F24D00"]
CLUSTER_COLORS = {i: COLOURS[i] for i in range(5)}
CLUSTER_COLORS[-1] = '#999999'

# =============================================================================
# LOAD DATA
# =============================================================================
clusters = pd.read_csv('processed-data/characterising-neighbourhoods/05_clusters.csv', 
                       dtype={'PollingAreaID': str})

# MANUAL SWING ASSIGNMENTS (create/edit valgsted_manual_swing.csv first)
manual_swing = pd.read_csv('valgsted_manual_swing.csv')
manual_swing['PollingAreaID'] = manual_swing['gruppe_id'].astype(str)
print(f"Manual: {manual_swing['swing_state'].sum()} swings / {len(manual_swing)} areas")

geo = gpd.read_file('processed-data/geodata/afstemningsomraader2021_CPH_FRB.geojson')

# Merged areas handling (your logic)
MERGED_AREAS = {
    '1010010': '1010008',
    '1010057': '1010014',
    '1470008': '1470007',
}

# =============================================================================
# PREPARE GDF WITH MANUAL SWING
# =============================================================================
# Geo PollingAreaID (your exact logic)
geo['PollingAreaID'] = (
    geo['kommunekode'].str[-3:] + 
    '0' + geo['afstemningsomraadenummer'].str.zfill(2)
)
geo['DataPollingAreaID'] = geo['PollingAreaID'].map(lambda x: MERGED_AREAS.get(x, x))

# Manual swing: map DataPollingAreaID
manual_swing['DataPollingAreaID'] = manual_swing['PollingAreaID'].map(
    lambda x: MERGED_AREAS.get(x, x)
)

# Merge clusters + manual swing
gdf = geo.merge(clusters[['PollingAreaID', 'Cluster_kMeans']], 
                left_on='DataPollingAreaID', right_on='PollingAreaID', 
                how='left', suffixes=('', '_cluster'))
gdf = gdf.merge(manual_swing[['DataPollingAreaID', 'swing_state']], 
                on='DataPollingAreaID', how='left')

gdf['swing_state'] = gdf['swing_state'].fillna(0).astype(int)
gdf['Cluster_kMeans'] = gdf['Cluster_kMeans'].fillna(-1).astype(int)

print(f"Final: {len(gdf)} areas, {gdf['swing_state'].sum()} swings")

# =============================================================================
# PLOT (unchanged)
# =============================================================================
gdf['facecolor'] = gdf['Cluster_kMeans'].map(CLUSTER_COLORS)
gdf['alpha'] = gdf['swing_state'].map(lambda x: 0.9 if x == 1 else 0.7)
gdf['edgecolor'] = gdf['swing_state'].map(lambda x: 'black' if x == 1 else 'white')
gdf['linewidth'] = gdf['swing_state'].map(lambda x: 1.5 if x == 1 else 1.5)

fig, (ax, leg_ax) = plt.subplots(1, 2, figsize=(12, 10), 
                                 gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05})

# Map plot
for idx, row in gdf.iterrows():
    gdf.iloc[[idx]].plot(ax=ax, color=row['facecolor'], alpha=row['alpha'],
                         edgecolor=row['edgecolor'], linewidth=row['linewidth'], zorder=1)
    
    if row['swing_state'] == 1:
        geom = gdf.iloc[idx].geometry
        if geom.geom_type == 'Polygon':
            coords = [(x, y) for x, y in zip(geom.exterior.coords.xy[0], geom.exterior.coords.xy[1])]
            patch = Polygon(coords, fill=False, hatch='///', 
                           edgecolor='black', linewidth=0, alpha=1, zorder=2)
            ax.add_patch(patch)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = [(x, y) for x, y in zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1])]
                patch = Polygon(coords, fill=False, hatch='///', 
                               edgecolor='black', linewidth=0, alpha=1, zorder=2)
                ax.add_patch(patch)

ax.axis('off')
ax.set_title(r'\textbf{Demographic Clusters Across Copenhagen and Frederiksberg: Swing Status}',
             fontsize=24, pad=20, ha='center', va='top')

# Legend
legend_elements = [Patch(facecolor=COLOURS[i], alpha=0.8, label=f'Cluster {i}') 
                   for i in range(5)]
legend_elements.append(Patch(facecolor='lightgray', hatch='///', edgecolor='black', 
                             alpha=1, label='Swing Area'))
leg_ax.legend(handles=legend_elements, fontsize=14, frameon=True, 
              loc='center', markerscale=1.2)
leg_ax.axis('off')

xlim = ax.get_xlim()
ax.set_xlim(xlim[0], xlim[1] * 0.994)

plt.tight_layout()
plt.savefig('cluster_map_with_swing_hatched.png', dpi=300, bbox_inches='tight')
plt.savefig('cluster_map_with_swing_hatched.pgf', bbox_inches='tight')
print("Saved: cluster_map_with_swing_hatched.png")