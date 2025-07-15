import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from tqdm import tqdm


# Funzione per trovare il miglior modello dal file di riepilogo riassunto_modelli.txt
def find_best_model_from_summary(filepath="output/stats/riassunto_modelli.txt"):

    best_f1_score = -1.0
    best_run_id = None
    best_model_params = {} # Dizionario per i parametri del modello migliore
    
    if not os.path.exists(filepath):
        print(f"Errore: Il file '{filepath}' non esiste.")
        return None, 0.0, {}

    print(f"Lettura del file: {filepath}")
    
    with open(filepath, 'r') as f:
        current_run_id = None
        current_config_params = {} 

        for line_num, line in enumerate(f):
            line = line.strip()

            if line.startswith("# --- INIZIO RUN:"):
                try:
                    current_run_id = line.split(":")[-1].strip().replace(" ---", "")
                    current_config_params = {} # Reset dei parametri 
                except IndexError:
                    print(f"Avviso: Formato 'INIZIO RUN' non valido alla riga {line_num + 1}: {line}")
                    current_run_id = None
                continue 
            
            # Cattura la riga Config: per estrarre tutti i parametri
            if line.startswith("Config:") and current_run_id:
                try:
                    config_str = line[len("Config:"):].strip()
                    parts = config_str.split(',')
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            # Tenta di convertire i valori numerici e booleani
                            try:
                                if value.lower() == 'true':
                                    current_config_params[key] = True
                                elif value.lower() == 'false':
                                    current_config_params[key] = False
                                else:
                                    current_config_params[key] = int(value)
                            except ValueError:
                                try:
                                    current_config_params[key] = float(value)
                                except ValueError:
                                    current_config_params[key] = value # Lascia come stringa se non è numero/bool
                    
                except (ValueError, IndexError) as e:
                    print(f"Avviso: Errore nel parsing della riga Config alla riga {line_num + 1} per Run ID '{current_run_id}': {e} - {line}")

            # Cerca la riga delle statistiche e assicurati di avere un run_id valido
            if line.startswith("Stats:") and current_run_id:
                try:
                    stats_str = line[len("Stats:"):].strip()
                    parts = stats_str.split(',')
                    
                    for part in parts:
                        if part.startswith("BEST_F1="):
                            f1_value_str = part.split('=', 1)[1].strip()
                            current_f1 = float(f1_value_str)
                            
                            if current_f1 > best_f1_score:
                                best_f1_score = current_f1
                                best_run_id = current_run_id
                                # Salva TUTTI i parametri di configurazione del modello migliore
                                best_model_params = current_config_params.copy() 
                            break
                except (ValueError, IndexError) as e:
                    print(f"Avviso: Errore nel parsing della riga Stats alla riga {line_num + 1} per Run ID '{current_run_id}': {e} - {line}")
                finally:
                    current_run_id = None # Reset per il prossimo blocco
                    current_config_params = {} # Reset anche dei parametri

    if best_run_id:
        print(f"\nAnalisi completata. Il modello migliore è:")
        print(f"  RUN_ID (Model Name): {best_run_id}")
        print(f"  Best F1-score: {best_f1_score:.4f}")
        print(f"  Parametri di Configurazione del modello migliore: {best_model_params}")
    else:
        print("\nNessun modello valido trovato nel file o file vuoto.")

    return best_run_id, best_f1_score, best_model_params


# Calcola la matrice di confusione pixel per pixel
def compute_pixel_confusion_matrix(loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Collecting pixels for confusion matrix"):
            x = x.to(device)
            y = y.to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(y.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    return cm


# Plotta la curva Precision-Recall
def plot_pr_curve(loader, model, device, title='Precision-Recall Curve sul Test Set'):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Collecting probabilities for PR curve"):
            x = x.to(device)
            y_flat = y.cpu().numpy().flatten()

            preds = torch.sigmoid(model(x)).cpu().numpy().flatten()

            all_probs.append(preds)
            all_targets.append(y_flat)

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()

    print(f"Precision-Recall AUC: {pr_auc:.4f}")
