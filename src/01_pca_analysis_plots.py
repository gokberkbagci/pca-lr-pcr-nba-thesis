import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Chargement du fichier CSV (Colab ou local) 
try:
    import google.colab
    from google.colab import files
    print("Mode Colab détecté. Veuillez importer le fichier manuellement.")
    uploaded = files.upload()
    nom_fichier = list(uploaded.keys())[0]
    df = pd.read_csv(nom_fichier)
except:
    print("Mode local détecté. Chargement depuis ../data/nba_player_stats.csv ...")
    chemin_fichier = os.path.join("..", "data", "nba_player_stats.csv")
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"Le fichier '{chemin_fichier}' est introuvable.")
    df = pd.read_csv(chemin_fichier)

# Sélection des variables numériques pertinentes
variables = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'MP']
X = df[variables].dropna()

# Attribution d’un rôle à chaque joueur
def attribuer_role(ligne):
    if ligne['PTS'] >= 15 and ligne['AST'] >= 3:
        return 'Attaque'
    elif ligne['STL'] >= 1.5 or ligne['BLK'] >= 1.5:
        return 'Défense'
    else:
        return 'Hybride'

df = df.loc[X.index]
df['Rôle'] = df.apply(attribuer_role, axis=1)

# Standardisation des données
scaler = StandardScaler()
X_standardisé = scaler.fit_transform(X)

# Application de l’ACP
pca = PCA()
composantes = pca.fit_transform(X_standardisé)
variance_expliquée = pca.explained_variance_ratio_

# Créer le dossier figures si nécessaire
os.makedirs("../figures", exist_ok=True)

# Création de la figure avec deux sous-graphiques
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 1. Graphique de la variance expliquée
barres = axs[0].bar(
    [f"Composante {i+1}" for i in range(len(variance_expliquée))],
    variance_expliquée * 100,
    color=plt.cm.viridis(np.linspace(0, 1, len(variance_expliquée)))
)
axs[0].set_xlabel("Composantes principales")
axs[0].set_ylabel("Variance expliquée (%)")
axs[0].set_title("Variance expliquée par chaque composante principale")
axs[0].tick_params(axis='x', rotation=45)
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

for barre in barres:
    hauteur = barre.get_height()
    axs[0].text(
        barre.get_x() + barre.get_width() / 2,
        hauteur + 0.5,
        f"{hauteur:.1f}%",
        ha='center',
        va='bottom',
        fontsize=10
    )

# 2. Projection des joueurs selon leurs rôles
pca_df = pd.DataFrame(composantes[:, :2], columns=['PC1', 'PC2'])
pca_df['Rôle'] = df['Rôle'].values
couleurs = {'Attaque': 'red', 'Défense': 'blue', 'Hybride': 'green'}

for role, couleur in couleurs.items():
    sous_ensemble = pca_df[pca_df['Rôle'] == role]
    axs[1].scatter(sous_ensemble['PC1'], sous_ensemble['PC2'],
                   label=role, color=couleur, alpha=0.6)

axs[1].set_xlabel("Composante Principale 1")
axs[1].set_ylabel("Composante Principale 2")
axs[1].set_title("Projection des joueurs selon leurs rôles (ACP)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("../figures/01_pca_analysis_plots.png")
plt.show()
