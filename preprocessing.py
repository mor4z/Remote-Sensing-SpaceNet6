import os
import rasterio
from rasterio.features import rasterize
import fiona
from shapely.geometry import shape
import numpy as np
import time

def preprocess_spacenet6_data(base_data_path):
    """
    Pre-elabora i dati SpaceNet 6 rasterizzando i file GeoJSON in maschere binarie.
    Le maschere rasterizzate verranno salvate nella sottocartella 'rasterized_masks'
    all'interno di base_data_path.
    """
    sar_intensity_dir = os.path.join(base_data_path, 'SAR-Intensity')
    geojson_buildings_dir = os.path.join(base_data_path, 'geojson_buildings')
    output_masks_dir = os.path.join(base_data_path, 'rasterized_masks')

    if not os.path.isdir(output_masks_dir):
        print(f"ERRORE: La cartella di output delle maschere '{output_masks_dir}' non esiste. Creala manualmente.")
        return

    print(f"La cartella di output per le maschere rasterizzate è: {output_masks_dir}")

    sar_files = [f for f in os.listdir(sar_intensity_dir) if f.endswith('.tif')]
    total_files = len(sar_files)
    
    if total_files == 0:
        print(f"Nessun file .tif trovato in {sar_intensity_dir}. Controlla il percorso.")
        return

    print(f"Trovati {total_files} file SAR. Inizio la rasterizzazione...")

    processed_count = 0
    start_time = time.time()
    
    for sar_filename in sar_files:
        sar_base_name = os.path.splitext(sar_filename)[0]
        geojson_base_name = sar_base_name.replace('_SAR-Intensity', '_Buildings')
        geojson_filename = f"{geojson_base_name}.geojson"
        
        sar_path = os.path.join(sar_intensity_dir, sar_filename)
        geojson_path = os.path.join(geojson_buildings_dir, geojson_filename)
        output_mask_path = os.path.join(output_masks_dir, f"{sar_base_name}.tif")

        processed_count += 1
        
        if processed_count % 50 == 0 or processed_count == 1 or processed_count == total_files:
            elapsed_time = time.time() - start_time
            print(f"Processing {processed_count}/{total_files} files. Elapsed time: {elapsed_time:.2f} seconds.")

        try:
            with rasterio.open(sar_path) as src:
                out_shape = src.shape
                out_transform = src.transform
                out_crs = src.crs

            geometries = []
            with fiona.open(geojson_path, 'r') as collection:
                for feature in collection:
                    if feature['geometry']:
                        geometries.append((shape(feature['geometry']), 255)) 

            mask = rasterize(
                geometries,
                out_shape=out_shape,
                transform=out_transform,
                fill=0,
                all_touched=False,
                dtype=np.uint8
            )

            profile = {
                'driver': 'GTiff',
                'height': out_shape[0],
                'width': out_shape[1],
                'count': 1,
                'dtype': 'uint8',
                'crs': out_crs,
                'transform': out_transform,
                'compress': 'lzw'
            }
            
            with rasterio.open(output_mask_path, 'w', **profile) as dst:
                dst.write(mask, 1)

        except Exception as e:
            print(f"Errore durante la rasterizzazione di {sar_filename} (cercato GeoJSON: {geojson_filename}): {e}")
            continue

    print("Rasterizzazione completata!")