import os

def save_ids_to_file(ids_list, filename, output_dir='./'):
    """
    Salva una lista di ID in un file di testo, uno per riga.
    """
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        for img_id in ids_list:
            f.write(f"{img_id}\n")
    print(f"ID salvati in: {filepath}")