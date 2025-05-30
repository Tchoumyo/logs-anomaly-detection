import numpy as np
import joblib # permet de sauvegarder les models entrainés dans des fichiers joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from constant import Constant
import matplotlib.pyplot as plt
from preprocessing import LogPreprocessor

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            n_jobs=-1, verbose=1
        )
        self.one_svm = OneClassSVM(verbose=1)

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
        fig ,(axi1,axi2) = plt.subplots(2,1)
        axi1.hist(score['isolation_forest'], bins=20, alpha=0.7,color='blue')
        axi1.set_title("Histogramme Isolation forest")
        axi2.hist(score['one_svm'], bins=20, alpha=0.7,color='red')
        axi2.set_title("Histogramme One class SVM")

        plt.xlabel("Scores d'anomalies")
        plt.ylabel("Fréquence")
        plt.tight_layout() # Jumeller les deux graphes
        plt.show()
        
if __name__=="__main__":
    df = pd.read_csv(Constant.LOGS_DATASET_FILE_NAME)
    
    preprocessor = LogPreprocessor()
    
    df_train,df_test = preprocessor.split_dataset(df)
    
    print("Prétraitement des données")
    x_train, df_train_engineered = preprocessor.fit_transform(df_train)
    x_test, df_test_engineered = preprocessor.fit_transform(df_test)
    
    detector = AnomalyDetector()
    
    print("Entrainement du modèle")
    detector.train_models(x_train)
    
    print("Evaluation du modèle")
    detector.evaluate_models(x_test)
    












