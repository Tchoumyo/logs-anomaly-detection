import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from constant import Constant

# Charger les données
logs_df = pd.read_csv(Constant.LOGS_DATASET_FILE_NAME)

# Ajouter un log personnalisé
nouveau_log = pd.DataFrame({
    'timestamp': ['2024-01-11 01:06:15.104135'],
    'user_ip': ['133.255.54.117'],
    'method': ['GET'],
    'end_point': ['/admin'],
    'status_code': [204],
    'response_time': [100]
})

# Standardisation
scaler = StandardScaler()
logs_df['response_time_scaled'] = scaler.fit_transform(logs_df[['response_time']])
nouveau_log['response_time_scaled'] = scaler.transform(nouveau_log[['response_time']])

# Sélection des features
features = ['status_code', 'response_time_scaled']
logs_features = logs_df[features]
logs_with_new = pd.concat([logs_features, nouveau_log[features]], ignore_index=True)

# Isolation Forest
model_if = IsolationForest(contamination=0.5, random_state=42)
model_if.fit(logs_with_new.iloc[:-1])
logs_with_new['anomaly_iforest'] = model_if.predict(logs_with_new)


# One-Class SVM
model_svm = OneClassSVM(nu=0.545)
model_svm.fit(logs_with_new.iloc[:-1])
logs_with_new['anomaly_svm'] = model_svm.predict(logs_with_new)


# Réduction de dimension pour visualisation
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(logs_with_new[features])
logs_with_new['x'] = reduced[:, 0]
logs_with_new['y'] = reduced[:, 1]

# Séparer selon les deux modèles
def split_by_model(df, col_name):
    normaux = df.iloc[:-1][df.iloc[:-1][col_name] == 1]
    anomalies = df.iloc[:-1][df.iloc[:-1][col_name] == -1]
    nouveau = df.iloc[-1]
    couleur = 'green' if nouveau[col_name] == 1 else 'black'
    print(col_name, "-->", anomalies.shape)
    
    return normaux, anomalies, nouveau, couleur

# Affichage
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, model_name, col in zip(
    axes,
    ['Isolation Forest', 'One-Class SVM'],
    ['anomaly_iforest', 'anomaly_svm']
):
    normaux, anomalies, nouveau_point, couleur = split_by_model(logs_with_new, col)
    ax.scatter(normaux['x'], normaux['y'], c='blue', label='Normal', alpha=0.5)
    ax.scatter(anomalies['x'], anomalies['y'], c='red', label='Anomalie', alpha=0.5)
    ax.scatter(nouveau_point['x'], nouveau_point['y'], c=couleur, label='Nouveau Log', s=150, edgecolor='yellow', linewidth=2)
    ax.set_title(f"t-SNE - {model_name}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(True)
    
    print(f"{model_name} score : ", anomalies.shape[0]/len(logs_with_new))


#plt.tight_layout()
#plt.show()

# Résultat
etat_iforest = "Normal" if logs_with_new.iloc[-1]['anomaly_iforest'] == 1 else "Anomalie"
etat_svm = "Normal" if logs_with_new.iloc[-1]['anomaly_svm'] == 1 else "Anomalie"

print(f"Nouveau log détecté par Isolation Forest : {etat_iforest}")
print(f"Nouveau log détecté par One-Class SVM : {etat_svm}")
