import os
import argparse
import json
import torch
from tqdm import tqdm
import sys

# Import-Pfade für Konfiguration und Metriken
from diffusion.utils.misc import read_config
sys.path.append("/home/hd/hd_hd/hd_om233/partially_removal/MasterThesis_Evaluation")
sys.path.append("/home/hd/hd_hd/hd_om233/partially_removal/")
from MasterThesis_Evaluation.evaluation_CLIP_2 import compute_clip
from MasterThesis_Evaluation.evaluation_cmmd import compute_cmmd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="/home/hd/hd_hd/hd_om233/partially_removal_individual_compression/PixArt-sigma/block_analysis_eval/test_1k_images/first_removal_stage_large_1k.yaml", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config = read_config(args.config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_list = config.block_list
    _base_image_path = config.save_path
    
    # Sicherstellen, dass das Log-Verzeichnis existiert
    os.makedirs(config.log_path, exist_ok=True)
    results_file = os.path.join(config.log_path, "evaluation_results_raw.txt")

    # Header für die Textdatei schreiben
    with open(results_file, "a") as f:
        f.write(f"\n--- Evaluation Run: {os.path.basename(args.config_path)} ---\n")
        f.write("Iteration | Block | CMMD (Raw) | CLIP Score\n")
        f.write("-" * 50 + "\n")



# ... (dein restlicher Code oben bleibt gleich)

    # Ergebnisse sammeln für das Ranking
    all_results = []

    for idx in tqdm(block_list):
        image_folder = os.path.join(_base_image_path, f"it_{0}", str(idx))
        if not os.path.exists(image_folder) or not os.listdir(image_folder):
            continue

        try:
            cmmd_val = compute_cmmd(config.ref_path, image_folder).item()
            clip_val = compute_clip(image_folder, config.prompt_path)
            
            # Für das Ranking speichern
            all_results.append({'idx': idx, 'cmmd': cmmd_val, 'clip': clip_val})
            
            # Tabellarischer Output
            with open(results_file, "a") as f:
                f.write(f"{0:^9} | {idx:^5} | {cmmd_val:10.6f} | {clip_val:10.6f}\n")
        except Exception as e:
            print(f"Error evaluating block {idx}: {e}")

    # --- RANKING LOGIK HINZUFÜGEN ---
    # Sortieren nach CMMD (niedrigster Wert ist am besten)
    sorted_results = sorted(all_results, key=lambda x: x['cmmd'])
    
    if sorted_results:
        best_block = sorted_results[0]['idx']
        
        with open(results_file, "a") as f:
            f.write(f"\nBest block: {best_block}\n")
            # Jedem Block seine Position im Ranking zuweisen
            for position, res in enumerate(sorted_results):
                # Wir suchen das Original-Ergebnis, um es in der ursprünglichen Reihenfolge auszugeben
                # oder wir geben es einfach sortiert aus. Dein Beispiel war nach Block-ID sortiert:
                pass 

            # Um exakt dein Output-Format zu erhalten (nach Block-ID sortiert, aber mit Rang):
            id_sorted = sorted(all_results, key=lambda x: x['idx'])
            # Mapping erstellen: Block-ID -> Rangplatz
            rank_mapping = {res['idx']: pos for pos, res in enumerate(sorted_results)}
            
            for res in id_sorted:
                f.write(f"Block: {res['idx']} CMMD: {res['cmmd']} total position: {rank_mapping[res['idx']]}\n")

    print(f"\nEvaluation beendet. Ergebnisse (inkl. Ranking) unter: {results_file}")