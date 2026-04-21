import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_benchmark(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"❌ Le fichier {csv_path} est introuvable.")
        return

    # 1. Lecture brute du fichier (Pandas va lire tout, y compris le texte à la fin)
    # On met low_memory=False pour éviter les warnings liés aux types mixtes
    df = pd.read_csv(csv_path, low_memory=False)

    # 2. Nettoyage intelligent : On force la colonne "Time (s)" en nombres.
    # Le texte de la fin (les métriques) deviendra NaN (Not a Number)
    df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')

    # 3. On supprime toutes les lignes où le temps est NaN (cela efface le texte de fin !)
    df = df.dropna(subset=['Time (s)'])

    # On s'assure que les BPM sont aussi bien des nombres
    df['Estimated_BPM'] = pd.to_numeric(df['Estimated_BPM'], errors='coerce')
    df['True_BPM'] = pd.to_numeric(df['True_BPM'], errors='coerce')

    # 4. Création du graphique professionnel
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['Time (s)'], df['True_BPM'], label="True BPM (Empatica E4)", color='black', linewidth=2)
    plt.plot(df['Time (s)'], df['Estimated_BPM'], label="Estimated BPM (Algorithme)", color='red', alpha=0.8, linewidth=2)

    plt.title(f"Analyse de Synchronisation : {os.path.basename(csv_path)}", fontsize=14, fontweight='bold')
    plt.xlabel("Temps (secondes)", fontsize=12)
    plt.ylabel("Fréquence Cardiaque (BPM)", fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    # Affichage de la fenêtre
    plt.show()

if __name__ == "__main__":
    # nom fichier CSV 
    FICHIER_CSV = "dataset/results/UBFC-rPPG-Set2-Realistic/POS/benchmark_results_subject1_POS.csv"
    
    plot_benchmark(FICHIER_CSV)