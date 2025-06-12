import numpy as np
import joblib # permet de sauvegarder les models entrainés dans des fichiers joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from constant import Constant
import matplotlib.pyplot as plt
from preprocessing import LogPreprocessor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE




class AnomalyDetector():
    def __init__(self):
        #self.element = element
        self.isolation_forest = IsolationForest(
            n_jobs=-1, verbose=1, contamination=0.2  
        )
        self.one_svm = OneClassSVM(verbose=1, nu=0.2)


    def train_models(self, X:np.ndarray):
        """
            Train isolation forest and one class SVM models
        """
        self.isolation_forest.fit(X)
        self.one_svm.fit(X)

        joblib.dump(self.isolation_forest,Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        joblib.dump(self.one_svm,Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

    def predict(self,X:np.ndarray)->dict[str,np.ndarray]:
        """
            Faire les prediction sur X en utilisant les modeles d'isolation forest ou one class svm 
        """
        isolation_forest: IsolationForest = joblib.load(Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        one_svm: OneClassSVM = joblib.load(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

        return {
            "isolation_forest" : isolation_forest.predict(X),
            "one_svm" : one_svm.predict(X)
        }


    def compute_anomaly_scores(self,X:np.ndarray) -> dict[str,np.ndarray]:
        """
            Calculer les scores d'anomalie sur X en utilisant les modeles d'isolation forest ou one class svm 
        """
        isolation_forest: IsolationForest = joblib.load(Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        one_svm: OneClassSVM = joblib.load(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

        return {
            "isolation_forest" : isolation_forest.decision_function(X),
            "one_svm" : one_svm.decision_function(X)
        }

    def evaluate_models(self,X:np.ndarray):
        """
            Evaluation des modeles d'isolation forest et de one class svm 
        """ 
        score = self.compute_anomaly_scores(X)
        # fig ,(axi1,axi2) = plt.subplots(2,1)
        # axi1.hist(score['isolation_forest'], bins=20, alpha=0.7,color='blue')
        # axi1.set_title("Histogramme Isolation forest")
        # axi2.hist(score['one_svm'], bins=20, alpha=0.7,color='red')
        # axi2.set_title("Histogramme One class SVM")

        # plt.xlabel("Scores d'anomalies")
        # plt.ylabel("Fréquence")
        # plt.tight_layout() # Jumeller les deux graphes
        # plt.show()
        
          # Réduction de dimension pour affichage
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        # svd = TruncatedSVD(n_components=2)
        # X_reduced = svd.fit_transform(X)
        X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

            # Scatter plot Isolation Forest
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=score['isolation_forest'], cmap='bwr', edgecolor='k', alpha=0.6)
        plt.title("Isolation Forest - Nuage de points (PCA)")
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.colorbar(label='Label prédiction (-1 = anomalie, 1 = normal)')
        plt.show()

        # Scatter plot One-Class SVM
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=score['one_svm'], cmap='bwr', edgecolor='k', alpha=0.6)
        plt.title("One-Class SVM - Nuage de points (PCA)")
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.colorbar(label='Label prédiction (-1 = anomalie, 1 = normal)')
        plt.show()
        
        
        
        # Calcul du pourcentage de point considéré comme outliers dans les modèles
        prediction = self.predict(X)
        
        outliers_isolation_forest_ration = np.mean(prediction["isolation_forest"]==-1)
        
        outliers_one_svm_ration = np.mean(prediction["one_svm"]==-1)
        
        print(f"ratio d'outliers pour l'isolation forest: {outliers_isolation_forest_ration:.2%}")
        print(f"ratio d'outliers pour one class svm: {outliers_one_svm_ration:.2%}")
        
if __name__=="__main__":
    df = pd.read_csv(Constant.LOGS_DATASET_FILE_NAME)
    
    preprocessor = LogPreprocessor()
    
    df_train,df_test = preprocessor.split_dataset(df)
    
    print("Prétraitement des données")
    x_train, df_train_engineered,_ = preprocessor.fit_transform(df_train)
    x_test, df_test_engineered,status_codes_test = preprocessor.fit_transform(df_test)
    
    detector = AnomalyDetector()
    
    print("Entrainement du modèle")
    detector.train_models(x_train)
    
    print("\nEvaluation du modèle")
    detector.evaluate_models(x_test)
    
    
    prediction = detector.predict(x_test)
    

    # for element in np.arange(0.1,0.6,0.1):
    #     detector = AnomalyDetector(element=element)
        
    #     print(f"Quand le paramètre={element}")
    #     prediction = detector.predict(x_test)
    #     Isolation_forest_outliers = np.where(prediction["isolation_forest"]==-1)
    #     one_svm_outliers = np.where(prediction["one_svm"]==-1)
    #     score_iso_forest = df_test_engineered['is_potential_anomalous'].iloc[Isolation_forest_outliers].value_counts(normalize=True)
    #     score_one_svm = df_test_engineered['is_potential_anomalous'].iloc[one_svm_outliers].value_counts(normalize=True)
    #     print("Score de l'isolation forest ", score_iso_forest)
    #     print("Score de one class svm ",score_one_svm)


#================================================================================================#

# Liste des codes d'erreur à surveiller
codes = ['400', '408', '500', '502', '503', '504','200','204','201']

#X_preprocessed,X, status_code = LogPreprocessor.fit_transform(df)
# Ajouter les prédictions et les codes dans un seul DataFrame pour analyse
df_test_results = df_test.copy()
df_test_results["status_code"] = status_codes_test
df_test_results["prediction_iso"] = prediction["isolation_forest"]
df_test_results["prediction_svm"] = prediction["one_svm"]

# Analyse : pourcentage d’anomalies détectées pour chaque code d’erreur
print("\n DÉTECTION DES CODES D'ERREUR COMME ANOMALIES :\n")
for code in codes:
    mask = df_test_results["status_code"] == code
    total = mask.sum()
    anomalies_iso = (df_test_results["prediction_iso"][mask] == -1).sum()
    anomalies_svm = (df_test_results["prediction_svm"][mask] == -1).sum()

    ratio_iso = anomalies_iso / total * 100 if total > 0 else 0
    ratio_svm = anomalies_svm / total * 100 if total > 0 else 0

    print(f"Code {code} ➤  {total} occurrences")
    print(f" - Isolation Forest : {anomalies_iso} anomalies détectées ({ratio_iso:.2f}%)")
    print(f" - One-Class SVM    : {anomalies_svm} anomalies détectées ({ratio_svm:.2f}%)\n")



