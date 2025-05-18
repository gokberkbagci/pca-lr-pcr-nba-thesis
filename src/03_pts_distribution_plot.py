import pandas as pd
import matplotlib.pyplot as plt
import os

# Chargement du fichier CSV (Colab ou local)
try:
    import google.colab
    from google.colab import files
    print("Mode Colab détecté. Veuillez importer le fichier manuellement.")
    uploaded = files.upload()
    nom_fichier = list(uploaded.keys())[0]
    df = pd.read_csv(nom_fichier)
except ImportError:
    print("Mode local détecté. Chargement depuis ../data/nba_player_stats.csv ...")
    chemin_fichier = os.path.join("..", "data", "nba_player_stats.csv")
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"Le fichier '{chemin_fichier}' est introuvable.")
    df = pd.read_csv(chemin_fichier)

# Sélection des 500 premières lignes et suppression des valeurs manquantes dans PTS
df = df.dropna(subset=["PTS"]).head(500)
pts_values = df["PTS"]

# Création de l'histogramme
plt.figure(figsize=(10, 6))
plt.hist(pts_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

# Lignes rouges pointillées pour indiquer les bornes [10, 25]
plt.axvline(x=10, color='red', linestyle='--', linewidth=2)
plt.axvline(x=25, color='red', linestyle='--', linewidth=2)

# Titres et légende
plt.title("Distribution des scores PTS (500 premiers joueurs)")
plt.xlabel("PTS")
plt.ylabel("Nombre de joueurs")
plt.legend(["Bornes [10, 25]"])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Sauvegarde dans le dossier figures
os.makedirs("../figures", exist_ok=True)
plt.savefig("../figures/03_pts_distribution_plot.png")
plt.show()


