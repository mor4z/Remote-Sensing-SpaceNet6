import torch
import dataset as dataset
from tqdm import tqdm

# Presa dal video di riferimento sulla UNET     
def train(train_loader, model, optimizer, criterion, scaler, scheduler, device):
    model.train()

    bar = tqdm(train_loader)
    total_loss = 0
    num_batches = 0
    for data, mask in bar:
        # Sposta i dati sulla GPU
        data = data.to(device)
        mask = mask.to(device).squeeze(dim = 1)
        # Scala i dati
        # Azzera i gradienti
        optimizer.zero_grad()
        
        # Forward pass
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(data)
            pred = pred.squeeze(dim = 1)
            # Calcolo della loss
            loss = criterion(pred, mask)
        
        # Backward pass
        scaler.scale(loss).backward()
        # Aggiornamento dei pesi
        scaler.step(optimizer)
        if(scheduler is not None):
            scheduler.step()
        scaler.update()
        
        # Somma della loss
        total_loss += loss.item()
        num_batches += 1
        # Aggiornamento della barra di progresso
        bar.set_description(f"Loss: {loss.item():.4f}")
    
    # Calcolo della loss media
    average_loss = total_loss / num_batches
    return average_loss