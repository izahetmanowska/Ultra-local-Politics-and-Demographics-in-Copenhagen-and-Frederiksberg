import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import json

def convert_to_geojson(input_file, output_file, simplify_tolerance=0.00001):
    """
    Convert the DAGI JSON files with WKT geometry to GeoJSON, using GeoPandas.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output GeoJSON file
        simplify_tolerance: Tolerance for simplification (higher = more simplified)
                          0.0001 is about 10 meters, 0.00001 is about 1 meter
    """
    df = pd.read_json(input_file)

    # filter to keep only the most detailed scale (1:10.000)
    df = df[df['skala'] == '1:10.000']

    columns_to_keep = [
        'navn',
        'afstemningsomraadenummer',
        'afstemningsstedNavn',
        'opstillingskredsLokalId',
        'kommuneLokalId',
        'geometri'
    ]
    df = df[columns_to_keep]
    
    # convert WKT strings to geometry objects
    df['geometry'] = df['geometri'].apply(wkt.loads) # type: ignore
    
    # drop original 'geometri' column
    df = df.drop('geometri', axis=1)
    
    # convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:25832')

    # convert to WGS84 (better for web maps)
    gdf = gdf.to_crs('EPSG:4326')

    # simplify geometries to reduce file size
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=simplify_tolerance, preserve_topology=True)

    # convert to geojson dict first
    geojson_dict = json.loads(gdf.to_json())
    
    # write with consistent formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(geojson_dict, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    print(f"Converted {len(gdf)} features to {output_file}")

if __name__ == "__main__":
    root = ''
    input_path = root + 'raw-data/geofiles/DAGI_V1_Afstemningsomraade_TotalDownload_json_Current_459.json'
    output_path = root + 'processed-data/geodata/afstemningsomraader2025.geojson'
    convert_to_geojson(input_path, output_path, simplify_tolerance=0.00001)