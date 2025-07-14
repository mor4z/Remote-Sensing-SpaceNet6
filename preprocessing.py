import os
import time
import rasterio
import fiona
from shapely.geometry import shape
from rasterio.features import rasterize
import numpy as np
import tqdm 
import cv2
import rasterio
import geojson
from PIL import Image, ImageDraw


def preprocess_spacenet6_data(base_data_path):
    """
    Pre-processamento dei dati rasterizzando i file GeoJSON in maschere binarie.
    Le maschere rasterizzate verranno salvate dentro 'rasterized_masks' all'interno di base_data_path.
    """
    sar_intensity_dir = os.path.join(base_data_path, 'SAR-Intensity')
    geojson_buildings_dir = os.path.join(base_data_path, 'geojson_buildings')
    output_masks_dir = os.path.join(base_data_path, 'rasterized_masks')

    # Controlla l'esistenza della cartella di output e la crea se necessario
    if not os.path.isdir(output_masks_dir):
        print(f"La cartella di output delle maschere '{output_masks_dir}' non esiste. La sto creando...")
        os.makedirs(output_masks_dir, exist_ok=True) # Aggiungi exist_ok=True per evitare errori se esistesse già
    else:
        print(f"La cartella di output per le maschere rasterizzate è: {output_masks_dir}")


    sar_files = [f for f in os.listdir(sar_intensity_dir) if f.endswith('.tif')]
    total_files = len(sar_files)
    
    if total_files == 0:
        print(f"Nessun file .tif trovato in {sar_intensity_dir}. Controlla il percorso.")
        return

    print(f"Trovati {total_files} file SAR. Inizio la rasterizzazione...")


    for sar_filename in tqdm.tqdm(sar_files, desc='Rasterizzazione maschere'):
        sar_base_name = os.path.splitext(sar_filename)[0]
        
        if '_SAR-Intensity' in sar_base_name:
            geojson_base_name = sar_base_name.replace('_SAR-Intensity', '_Buildings')
        else:
            geojson_base_name = sar_base_name + '_Buildings' # Esempio: aggiunge _Buildings se non trova SAR-Intensity
           

        geojson_filename = f"{geojson_base_name}.geojson"
        
        sar_path = os.path.join(sar_intensity_dir, sar_filename)
        geojson_path = os.path.join(geojson_buildings_dir, geojson_filename)
        output_mask_path = os.path.join(output_masks_dir, f"{sar_base_name}.tif") # Il formato output desiderato è .tif

       
        try:
            # Apre l'immagine SAR per ottenere le sue proprietà georeferenziate
            with rasterio.open(sar_path) as src:
                out_shape = src.shape
                out_transform = src.transform
                out_crs = src.crs

            # Verifica se il file GeoJSON esiste
            if not os.path.exists(geojson_path):
                tqdm.tqdm.write(f"Skipping {sar_filename}: GeoJSON file not found at {geojson_path}")
                continue # Salta questo file e passa al successivo

            geometries = []
            # Apre il file GeoJSON con fiona
            with fiona.open(geojson_path, 'r') as collection:
                for feature in collection:
                    if feature['geometry']: # Assicurati che la feature abbia una geometria
                        geometries.append((shape(feature['geometry']), 255)) # Assegna il valore 255 ai poligoni

            # Rasterizza le geometrie sulla maschera
            mask = rasterize(
                geometries,
                out_shape=out_shape,
                transform=out_transform,
                fill=0, # Valore per il background (nero)
                all_touched=False, # Imposta su True se vuoi che tutti i pixel toccati dal bordo siano inclusi
                dtype=np.uint8 # Tipo di dato per la maschera (0-255)
            )

            # Prepara il profilo per il salvataggio della maschera TIFF
            profile = {
                'driver': 'GTiff',
                'height': out_shape[0],
                'width': out_shape[1],
                'count': 1, # Un singolo canale per la maschera binaria
                'dtype': 'uint8',
                'crs': out_crs, # Mantieni lo stesso CRS dell'immagine SAR
                'transform': out_transform, # Mantieni la stessa trasformazione dell'immagine SAR
                'compress': 'lzw' # Compressione per ridurre la dimensione del file
            }
            
            # Salva la maschera rasterizzata
            with rasterio.open(output_mask_path, 'w', **profile) as dst:
                dst.write(mask, 1) # Scrivi la maschera nel primo canale

        except Exception as e:
            tqdm.tqdm.write(f"Errore durante la rasterizzazione di {sar_filename} (cercato GeoJSON: {geojson_filename}): {e}")
            continue

    print("\nRasterizzazione completata!") # Stampa un messaggio finale dopo la barra di progresso


# Altro processo per la realizzazione delle maschere
def create_masks(image_folder, geojson_folder):
    # Creo la cartella "masks" de non dovesse esistere
    masks_folder = os.path.join(image_folder, '../masks')
    os.makedirs(masks_folder, exist_ok=True)

    # Lista di tutte le immagini nella cartella
    image_files = os.listdir(image_folder)

    for image_file in tqdm.tqdm(image_files, desc='Creating masks'):
        # Percorso dell'immagine
        image_path = os.path.join(image_folder, image_file)
        image = rasterio.open(image_path)
        # File geojson corrispondente
        geojson_path = image_path.strip().replace("/PS-RGB/", "/geojson_buildings/").replace("PS-RGB","Buildings").replace(".tif", ".geojson")

        # Creazione della maschera
        mask = geojson_to_mask(image, geojson_path)

        # Salvo le maschere nella cartella "masks"
        mask_file = image_file.replace('.tif', '.png')
        mask_path = os.path.join(masks_folder, mask_file)
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)


def geojson_to_mask(image, geojson_path):
    # Leggi il file GeoJSON
    with open(geojson_path, 'r') as f:
        geojson_data = geojson.load(f)

    # Crea un'immagine binaria vuota
    mask = Image.new('L', (image.shape[0], image.shape[1]), 0)
    draw = ImageDraw.Draw(mask)

    # Disegna i poligoni sulla maschera
    for feature in geojson_data['features']:
        geom = shape(feature['geometry'])
        if geom.geom_type == 'Polygon':
            coords = get_coords(image.transform, geom.exterior.coords)
            draw.polygon(coords, fill=255)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords = get_coords(image.transform, poly.exterior.coords)
                draw.polygon(coords, fill=255)

    # Converti l'immagine PIL in un array NumPy
    mask_np = np.array(mask)
    binary_mask = (mask_np > 0).astype(np.uint8)
    return binary_mask

def get_coords(transform, coords):
    new_coords = []
    for coord in coords:
        print(coord)
        x, y = coord[:2]
        px, py = ~transform * (x, y)
        new_coords.append((int(px), int(py)))
    return new_coords