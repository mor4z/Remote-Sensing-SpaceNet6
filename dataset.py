# Remote-Sensing-SpaceNet6/dataset.py

import os
import rasterio
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SpaceNet6Dataset(Dataset):
    # QUESTA RIGA DEVE INCLUDERE 'image_ids_list=None'
    def __init__(self, base_data_path, image_ids_list=None, transform=None): 
        """
        Inizializza il dataset SpaceNet 6.

        Args:
            base_data_path (str): Il percorso alla cartella 'AOI_11_Rotterdam'.
            image_ids_list (list, optional): Lista degli ID delle immagini da includere in questo dataset.
                                             Se None, carica tutti gli ID dalla cartella SAR-Intensity.
            transform (callable, optional): Trasformazioni aggiuntive da applicare.
        """
        self.sar_intensity_dir = os.path.join(base_data_path, 'SAR-Intensity')
        self.rasterized_masks_dir = os.path.join(base_data_path, 'rasterized_masks')
        self.transform = transform 

        if image_ids_list is not None:
            self.image_ids = image_ids_list
        else:
            self.image_ids = [
                os.path.splitext(f)[0]
                for f in os.listdir(self.sar_intensity_dir)
                if f.endswith('.tif') and os.path.isfile(os.path.join(self.sar_intensity_dir, f)) 
            ]

        if not self.image_ids:
            raise RuntimeError(f"Nessun ID immagine valido trovato o fornito per il dataset in {self.sar_intensity_dir}.")

        print(f"Dataset inizializzato con {len(self.image_ids)} coppie immagine-maschera.")

        self.to_tensor = transforms.ToTensor() 

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # ... (il resto del metodo __getitem__ è corretto) ...
        img_id = self.image_ids[idx]

        sar_filename = f"{img_id}.tif"
        mask_filename = f"{img_id}.tif" 

        sar_path = os.path.join(self.sar_intensity_dir, sar_filename)
        mask_path = os.path.join(self.rasterized_masks_dir, mask_filename)

        with rasterio.open(sar_path) as src_sar:
            image_np = src_sar.read().astype(np.float32) 
            image_tensor = torch.from_numpy(image_np) 

        with rasterio.open(mask_path) as src_mask:
            mask_np = src_mask.read(1).astype(np.float32) 
            mask_tensor = self.to_tensor(Image.fromarray(mask_np.astype(np.uint8)))

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor.squeeze(0)