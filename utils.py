import os
import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import torchvision.utils
from shapely.geometry import shape
import tqdm

def save_ids_to_file(ids_list, filename, output_dir='./'):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        for img_id in ids_list:
            f.write(f"{img_id}\n")
    print(f"ID salvati in: {filepath}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # check sulla GPU
        torch.cuda.manual_seed_all(seed) 
    print(f"Seed impostato su {seed} per la riproducibilità.")


# Don't start the training process without checking CUDA availability
def set_cuda_and_seed():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    if device != torch.device('cuda'):
        sys.exit("CUDA not available.  Exiting.")
    set_seed(42) # Set the seed to 42, which is the answer to everything
    return device


# Get the mean and standard deviation of the dataset for normalization transforms
def get_mean_std(path_to_train_data):
    mean = np.zeros(3)
    std = np.zeros(3)
    samples = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    with open(path_to_train_data) as f:
        image_paths = f.read().splitlines()
    
    progress_bar = tqdm.tqdm(total=len(image_paths), desc='Calculating mean and std')
    
    for image_path in image_paths:
        with rasterio.open(image_path) as dataset:
            image = dataset.read()
            image = torch.from_numpy(image).to(device)
            
            for i in range(3):
                mean[i] += torch.mean(image[i].float()).item()
                std[i] += torch.std(image[i].float()).item()
        
        samples += 1
        progress_bar.update(1)
    
    progress_bar.close()
    
    mean /= samples
    std /= samples

    return mean, std

def visualize_image(image, mask, prediction = None, save_path = None):
    # Funzione che stampa l'immagine e la maschera
    image = np.array(image).transpose(1,2,0)
    mask = np.array(mask).squeeze()
    plt.figure(figsize=(10, 10))
    if(prediction is not None):
        plt.subplot(1, 3, 1)
    else:
        plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Imamgine")
    if(prediction is not None):
        plt.subplot(1, 3, 2)
    else:
        plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.title("Maschera")
    if(prediction is not None):
        plt.subplot(1, 3, 3)
        plt.imshow(prediction, cmap='gray')
        plt.axis('off')
        plt.title("Predizione")
    if(save_path is not None):
        plt.savefig(save_path)
    plt.show()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_evals(data_loader, model, criterion, device, save_predictions=False, output_path=None):
    # Funzione che calcola accuracy, precision, recall, e F1 score di un modello
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    model.eval()
    total_loss = 0
    num_batches = 0
    if(save_predictions):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("Predizioni salvate in:", output_path)
    with torch.no_grad():
        for idx, (data, mask) in enumerate(data_loader):
            data = data.to(device)
            mask = mask.to(device).squeeze(dim=1)
            out  = model(data).squeeze(dim=1)
            loss = criterion(out, mask)
            total_loss += loss.item()
            num_batches += 1
            pred = torch.sigmoid(out)
            pred = (pred > 0.5).float()

            
            true_positives += ((pred == 1) & (mask == 1)).sum()
            false_positives += ((pred == 1) & (mask == 0)).sum()
            false_negatives += ((pred == 0) & (mask == 1)).sum()
            true_negatives += ((pred == 0) & (mask == 0)).sum()
            if save_predictions and idx % 10 == 0:
                torchvision.utils.save_image(pred.unsqueeze(dim=1), os.path.join(output_path, f"{idx} - predictions.png"))
                torchvision.utils.save_image(mask.unsqueeze(dim=1), os.path.join(output_path, f"{idx} - masks.png"))
    
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives + 1e-8)
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    model.train()
    return total_loss/num_batches, precision.item(), recall.item(), f1.item(), accuracy.item()

def save_checkpoint(state, filename="model_checkpoint.pth"):
    print("Salvataggio del checkpoint")
    torch.save(state, filename)

def load_checkpoint(name, model = None, optimizer = None, criterion = None):
    print("Caricamento del checkpoint")
    checkpoint = torch.load(name)
    if model is not None:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if criterion is not None:
        criterion.load_state_dict(checkpoint['loss'])
    return checkpoint['history']
    
def get_random_image(data_loader, model, device):
    for data, mask in data_loader:
        data = data.to(device)
        mask = mask.to(device).squeeze(dim=1)
        pred = torch.sigmoid(model(data).squeeze(dim=1))
        pred = (pred > 0.5).float()
        idx = random.randint(0, data.size(0) - 1)
        return data[idx], mask[idx], pred[idx]


# if __name__ == "__main__":
#     print("Calcolo della media e della deviazione standard del dataset")
# 
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#     print(f"La cartella dello script è: {script_directory}")
# 
#     mean, std = get_mean_std("split/train.txt")
#     print("Mean:", mean)
#     print("Standard Deviation:", std)