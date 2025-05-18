import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
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


# Sélection des variables explicatives et de la variable cible
variables = ['AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'MP']
df_clean = df[variables + ['PTS']].dropna()
X = df_clean[variables]
y = df_clean['PTS']

# Standardisation des variables explicatives
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Régression linéaire multiple
modele = LinearRegression()
modele.fit(X_scaled, y)
y_pred = modele.predict(X_scaled)

# Évaluation du modèle
r2 = r2_score(y, y_pred)
mco = mean_squared_error(y, y_pred)
print(f"R² (coefficient de détermination) : {r2:.3f}")
print(f"Erreur quadratique moyenne (EQM) : {mco:.2f}")

# Création du dossier pour enregistrer la figure
os.makedirs("../figures", exist_ok=True)

# Visualisation : PTS réels vs prédits + résidus
plt.figure(figsize=(10, 7))
plt.title("Régression linéaire : PTS réel vs PTS prédit")
plt.xlabel("Valeurs réelles de PTS")
plt.ylabel("Valeurs prédites de PTS")
plt.grid(True)

# Tracer les lignes de résidus et les points de prédiction
n = 500
for i in range(min(n, len(y))):
    plt.plot([y.iloc[i], y.iloc[i]], [y_pred[i], y.iloc[i]], color='red', alpha=0.3)
    plt.scatter(y.iloc[i], y_pred[i], color='gold', edgecolor='black', s=50)

# Ligne de prédiction parfaite
lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
plt.plot(lims, lims, linestyle='--', color='black', label='Prédiction parfaite')
plt.legend()
plt.tight_layout()
plt.savefig("../figures/02_mlr_analysis.png")
plt.close()

# Visualisation 2 : Bar chart des coefficients
plt.figure(figsize=(10, 6))
plt.barh(variables, modele.coef_, color='teal', edgecolor='black')
plt.axvline(x=0, color='grey', linestyle='--')
plt.title("Coefficients des variables explicatives (MLR)")
plt.xlabel("Valeur du coefficient")
plt.tight_layout()
plt.savefig("../figures/02_mlr_coefficients.png")
plt.close()

# Affichage terminal (facultatif mais utile pour vérification rapide)
print("\nTerme constant (β₀) :")
print(f"{modele.intercept_:.4f}")

print("\nCoefficients des variables explicatives (β₁ à β₈) :")
for var, coef in zip(variables, modele.coef_):
    print(f"{var} : {coef:.4f}")


