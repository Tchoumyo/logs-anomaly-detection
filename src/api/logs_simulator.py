import pandas as pd
import requests
import sys
import os

sys.path.append(os.getcwd())
from src.api.main import LogEntry
from logs.constant import Constant

class LogSimulator:
    def __init__(self, log_file_path:str, api_url:str, batch_size: int):
        self.log_file_path = log_file_path
        self.api_url = api_url
        self.batch_size = batch_size

    # Methode qui prend un log en paramètre a fin de l'ajouter dans notre API
    
    def send_log(self, log_entry:LogEntry):
        """Methode qui prend un log en paramètre a fin de l'ajouter dans notre API
        """
   
   # La librairie request permet d'envoyer des requêtes HTTP vers une API via un code python     
        try:  
            response = requests.post( url=self.api_url, 
                                json={
                                    "timestamp": log_entry.timestamp,
                                    "user_ip": log_entry.user_ip,
                                    "method": log_entry.method,
                                    "status_code": log_entry.status_code,
                                    "end_point": log_entry.end_point,
                                    "response_time": log_entry.response_time
                                })
            print(f"log {response.json()} ajouté avec succès")
        except Exception as e:
            print(f"echec de l'envoie de log: {e}")
            
    
    def simulate(self):
        """Lire le fichier de log et appel la methode send_log pour ajouter chaque log dans l'API
        """
        log_df = pd.read_csv(self.log_file_path)
        log_df_batch = log_df.sample(self.batch_size) # selectionner un certain nombre de log de facon aléatoire
        
        print("Debut de la simulation......")
        for idx, log_entry in log_df_batch.iterrows():
            self.send_log(log_entry=LogEntry(
                timestamp=log_entry.timestamp,
                user_ip=log_entry.user_ip,
                method=log_entry.method,
                status_code=str(log_entry.status_code),
                end_point=log_entry.end_point,
                response_time=log_entry.response_time
            ))
        print("Fin de la simulation......")
        
if __name__=="__main__":
    log_simulator = LogSimulator(
        log_file_path = Constant.LOGS_DATASET_FILE_NAME,
        api_url= "http://127.0.0.1:8000/log",
        batch_size=5000
    )
    
    log_simulator.simulate()