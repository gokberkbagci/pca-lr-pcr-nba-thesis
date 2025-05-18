import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
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

# Sélection et standardisation des variables explicatives
variables = ['AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'MP']
df_clean = df[variables + ['PTS']].dropna()
X = df_clean[variables]
y = df_clean['PTS']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCR sur les 500 premiers joueurs
X_pcr = X_scaled[:500]
y_pcr = y.iloc[:500]

# Application de l'ACP
pca = PCA()
X_pca = pca.fit_transform(X_pcr)

# Sélection des composantes principales expliquant ~94% de la variance
variance_cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(variance_cumsum >= 0.94) + 1
print(f"{n_components} composantes principales expliquent environ 94% de la variance.")

# Réduction des dimensions
X_pca_reduit = X_pca[:, :n_components]

# Régression sur composantes principales (PCR)
model_pcr = LinearRegression()
model_pcr.fit(X_pca_reduit, y_pcr)
y_pred_pcr = model_pcr.predict(X_pca_reduit)

# Évaluation du modèle
r2_pcr = r2_score(y_pcr, y_pred_pcr)
mco_pcr = mean_squared_error(y_pcr, y_pred_pcr)
print(f"R² (PCR) : {r2_pcr:.3f}")
print(f"Erreur quadratique moyenne (MCO) : {mco_pcr:.2f}")

# Création du dossier figures
os.makedirs("../figures", exist_ok=True)

# Visualisation : PTS réel vs prédit
plt.figure(figsize=(10, 7))
plt.title("Régression PCR : PTS réel vs PTS prédit")
plt.xlabel("Valeurs réelles de PTS")
plt.ylabel("Valeurs prédites de PTS")
plt.grid(True)

for i in range(len(y_pcr)):
    plt.plot([y_pcr.iloc[i], y_pcr.iloc[i]], [y_pred_pcr[i], y_pcr.iloc[i]], color='red', alpha=0.3)
    plt.scatter(y_pcr.iloc[i], y_pred_pcr[i], color='gold', edgecolor='black', s=50)

lims = [min(y_pcr.min(), y_pred_pcr.min()), max(y_pcr.max(), y_pred_pcr.max())]
plt.plot(lims, lims, linestyle='--', color='black', label='Prédiction parfaite')

plt.legend()
plt.tight_layout()
plt.savefig("../figures/04_pcr_residuals_plot.png")
plt.show()

