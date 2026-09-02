import geopandas as gpd

# Input and output file paths
root = ''
input_path = root + 'processed-data/geodata/afstemningsomraader2025.geojson'
output_path = root + 'processed-data/geodata/afstemningsomraader2025_CPH_FRB.geojson'

# Kommune IDs for Copenhagen and Frederiksberg
copenhagen_id = 389103
frederiksberg_id = 389104

# Read the GeoJSON file as a geopandas dataframe
gdf = gpd.read_file(input_path)

# Filter for Copenhagen and Frederiksberg
filtered_gdf = gdf[gdf['kommuneLokalId'].isin([copenhagen_id, frederiksberg_id])].copy()

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