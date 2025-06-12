import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from constant import Constant

class AnomalyComparator:
    def __init__(self):
        self.isolation_forest = IsolationForest(n_jobs=-1, random_state=42, contamination=0.5)
        self.one_svm = OneClassSVM(nu=0.545)

    def train_models(self, X: np.ndarray):
        self.isolation_forest.fit(X)
        self.one_svm.fit(X)

        joblib.dump(self.isolation_forest, Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        joblib.dump(self.one_svm, Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        isolation_forest: IsolationForest = joblib.load(Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        one_svm: OneClassSVM = joblib.load(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

        return {
            "anomaly_iforest": isolation_forest.predict(X),
            "anomaly_svm": one_svm.predict(X)
        }

    def visualize(self, X: np.ndarray, predictions: dict[str, np.ndarray]):
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(X)

        df_viz = pd.DataFrame(X, columns=['status_code', 'response_time_scaled'])
        df_viz['x'] = reduced[:, 0]
        df_viz['y'] = reduced[:, 1]
        df_viz['anomaly_iforest'] = predictions['anomaly_iforest']
        df_viz['anomaly_svm'] = predictions['anomaly_svm']

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, model_name, col in zip(
            axes,
            ['Isolation Forest', 'One-Class SVM'],
            ['anomaly_iforest', 'anomaly_svm']
        ):
            normal = df_viz[df_viz[col] == 1]
            anomalies = df_viz[df_viz[col] == -1]

            ax.scatter(normal['x'], normal['y'], c='blue', label='Normal', alpha=0.5)
            ax.scatter(anomalies['x'], anomalies['y'], c='red', label='Anomalie', alpha=0.5)
            ax.set_title(f"t-SNE - {model_name}")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend()
            ax.grid(True)

            print(f"{model_name} - Pourcentage d'anomalies : {len(anomalies)/len(df_viz):.2%}")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv(Constant.LOGS_DATASET_FILE_NAME)

    scaler = StandardScaler()
    df['response_time_scaled'] = scaler.fit_transform(df[['response_time']])

    features = ['status_code', 'response_time_scaled']
    x_train = df[features].to_numpy()

    comparator = AnomalyComparator()
    print("Entraînement et sauvegarde des modèles")
    comparator.train_models(x_train)

    print("\nPrédictions en cours")
    predictions = comparator.predict(x_train)

    print("\nVisualisation des résultats")
    comparator.visualize(x_train, predictions)
