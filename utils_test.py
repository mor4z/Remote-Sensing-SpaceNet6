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
        for x, y in tqdm(loader, desc="Calcolo della matrice di confusione..."):
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
        for x, y in tqdm(loader, desc="Calcolo della curva Precision-Recall..."):
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

# Stampa un grafico riassuntivo delle statistiche dei modelli
def plot_model_stats_bar_chart(filepath="output/stats/riassunto_modelli.txt", output_dir="output/stats"):
    if not os.path.exists(filepath):
        print(f"Errore: Il file '{filepath}' non esiste. Impossibile generare il grafico.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    model_stats = {} # Dizionario per contenere le statistiche per ogni modello
    current_run_id = None

    print(f"Lettura del file '{filepath}' per generare il grafico delle statistiche...")

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()

            if line.startswith("# --- INIZIO RUN:"):
                try:
                    current_run_id = line.split(":")[-1].strip().replace(" ---", "")
                except IndexError:
                    print(f"Avviso: Formato 'INIZIO RUN' non valido alla riga {line_num + 1}: {line}")
                    current_run_id = None
                continue
            
            if line.startswith("Stats:") and current_run_id:
                try:
                    stats_str = line[len("Stats:"):].strip()
                    parts = stats_str.split(',')
                    
                    stats_dict = {}
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip() 
                            value = value.strip() 
                            try:
                                stats_dict[key] = float(value) # Prova a convertire in float
                            except ValueError:
                                stats_dict[key] = value # Lascia come stringa se non è un numero
                    
                    model_stats[current_run_id] = stats_dict
                except (ValueError, IndexError) as e:
                    print(f"Avviso: Errore nel parsing della riga Stats alla riga {line_num + 1} per Run ID '{current_run_id}': {e} - {line}")
                finally:
                    current_run_id = None 

    if not model_stats:
        print("Nessuna statistica valida trovata nel file. Impossibile generare il grafico.")
        return

    # Estrai delle metriche di interesse
    first_model_key = list(model_stats.keys())[0]
    metrics = [key for key in model_stats[first_model_key].keys() if isinstance(model_stats[first_model_key][key], (int, float))]
    
    print(metrics)
    
    metrics_to_plot = ['TRAIN_LOSS', 'VAL_LOSS', 'BEST_F1', 'ACCURACY'] 
    actual_metrics_to_plot = [m for m in metrics_to_plot if m in metrics]

    if not actual_metrics_to_plot:
        print("Nessuna metrica numerica valida da plottare trovata. Verificare il formato del file 'riassunto_modelli.txt'.")
        return

    # Preparazione dei dati per il grafico
    model_names = list(model_stats.keys())
    num_models = len(model_names)
    num_metrics = len(actual_metrics_to_plot)

    if num_models == 0 or num_metrics == 0:
        print("Non ci sono dati sufficienti per generare il grafico.")
        return

    bar_width = 0.15 
    spacing = 0.05 # 
    index = np.arange(num_models) * (num_metrics * bar_width + spacing) 

    plt.figure(figsize=(num_models * 2 + num_metrics * 1.5, 8)) 

    # Genera le barre per ogni metrica
    for i, metric in enumerate(actual_metrics_to_plot):
        values = [model_stats[model_name].get(metric, 0.0) for model_name in model_names]
        plt.bar(index + i * bar_width, values, bar_width, label=metric)

    plt.xlabel('Modelli')
    plt.ylabel('Valore della Metrica')
    plt.title('Comparazione delle Performance dei Modelli (Statistiche di Test)')
    plt.xticks(index + (num_metrics - 1) * bar_width / 2, model_names, rotation=45, ha='right')
    plt.legend(title="Metriche")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adatta il layout per evitare sovrapposizioni

    plot_filename = os.path.join(output_dir, "grafico_riassuntivo.png")
    plt.savefig(plot_filename)
    print(f"Grafico delle statistiche salvato in: {plot_filename}")
    plt.show()