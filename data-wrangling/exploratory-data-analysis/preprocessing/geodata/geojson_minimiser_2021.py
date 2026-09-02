import geopandas as gpd

# Input and output file paths
input_path = 'processed-data/geodata/afstemningsomraader2021.geojson'
output_path = 'processed-data/geodata/afstemningsomraader2021_CPH_FRB.geojson'

# Kommune IDs for Copenhagen and Frederiksberg
copenhagen_id = "0101"
frederiksberg_id = "0147"

# Read the GeoJSON file as a geopandas dataframe
gdf = gpd.read_file(input_path)

# Filter for Copenhagen and Frederiksberg
filtered_gdf = gdf[gdf['kommunekode'].isin([copenhagen_id, frederiksberg_id])].copy()

# Convert to a projected CRS for accurate area calculation (UTM Zone 32N for Denmark)
filtered_gdf_projected = filtered_gdf.to_crs(epsg=25832)

# Calculate area in square meters
filtered_gdf['areaSquareMetres'] = filtered_gdf_projected.geometry.area

# Round to 2 decimal places
filtered_gdf['areaSquareMetres'] = filtered_gdf['areaSquareMetres'].round(2)

# Save to new GeoJSON file
filtered_gdf.to_file(output_path, driver='GeoJSON')

print(f"Filtered GeoJSON created successfully!")
print(f"Total features in original file: {len(gdf)}")
print(f"Features in Copenhagen and Frederiksberg: {len(filtered_gdf)}")
print(f"\nArea statistics (sq meters):")
print(f"  Minimum: {filtered_gdf['areaSquareMetres'].min():,.2f}")
print(f"  Maximum: {filtered_gdf['areaSquareMetres'].max():,.2f}")
print(f"  Mean: {filtered_gdf['areaSquareMetres'].mean():,.2f}")
print(f"\nOutput saved to: {output_path}")