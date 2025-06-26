import json
import time
import pandas as pd
from logs.constant import Constant
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from celery import Celery
import redis
import os
from pathlib import Path
from logs.Tests import AnomalyComparator


REDIS_LOG_KEY = "log_buffer" # Définition de la variable globale
THRESHOLD = 1000
REDIS_COUNTER_KEY ="log_counter"
REDIS_PREFIX_PREDICTION_KEY = "prediction"

# Uvicorn permet à Fastapi de gérer les requêtes HTTP
# pydantic une librairie de fastapi qui permet de denifir nos modèles sous formes de class





app = FastAPI(
            title="Logs Anomalies detection API",
            description="API for detecting anomalies in logs data",
            version="1.1.0"
    ) # initialisation de l'application


try:
    redis_client = redis.Redis(host="redis", port=6379, db=0) # chargé d'interagir avec la db redis, broker de celery, il transmet le message et distribu les tâches
    print("Redis est lancé avec succès")

except redis.exceptions.ConnectionError as e:
    print(" Redis n'est pas lancé. L'application démarre sans Redis.")
    redis_client = None

# Initialisation de celery, celery est une technologie qui permet d'executer les tâches en background
celery_app = Celery("anomalies detection", broker="redis://redis:6379/0", backend="redis://redis:6379/0")


class LogEntry(BaseModel): # represente la structure d'un log
    timestamp: str
    user_ip: str
    method: str
    status_code: int
    end_point: str
    response_time: float
   
# Format de sorti

class PredictionResponse(BaseModel):
    isolation_forest_prediction: List[int]
    one_class_prediction: List[int]
    anomalies_detected: int
    total_entry: int
    logs_id: List[int]
    
@celery_app.task(name="Process_batch_prediction")
def Process_batch_prediction(log_data_json):
    try:
        log_entries = json.loads(log_data_json)
        
        df = pd.DataFrame(log_entries)
        
        detector = AnomalyComparator()
        
        scores = detector.predict(df)
        
        prediction = detector.predict(df)
        
        isolation_forest_preds = prediction['anomaly_iforest'].tolist()
        one_class_svm_preds = prediction['anomaly_svm'].tolist()
        anomalies_count = sum(1 for pred in isolation_forest_preds if pred==-1)
        result = {
            'isolation_forest_preds': isolation_forest_preds,
            'one_class_svm_preds':one_class_svm_preds,
            'scores':{
                'isolation_forest_score': scores['isolation_forest_score'].tolist(),
                'one_svm_score': scores['one_svm_score'].tolist()
            },
            'anomalies_count': anomalies_count,
            'timestamp': time.time()
        }
        
        prediction_id = int(time.time())
        
        # Enregistrer result dans redis
        redis_client.setex(
            f"{REDIS_PREFIX_PREDICTION_KEY}{prediction_id}",
            86400, # temps en seconde
            json.dumps(result) # toujours sérialiser avant de mettre dans redis
        )
        
        redis_client.lpush("prediction_list", prediction_id)
        
        return {
            "message": f"{prediction_id} prediction completed",
            "status": "success"
        }
    except Exception as e:
        return{
            "message": f"{str(e)}",
            "status": "error"
        }

# Définition de la route
@app.get("/")
def root():
    
    return {
        "message" : "API is ready !!!"
    }
    
@app.get("/status_model")
def check_model():
    isolation_forest_exist = os.path.exists(Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
    one_class_svm_exist = os.path.exists(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)
    
    if not (isolation_forest_exist and one_class_svm_exist):
        raise HTTPException(
            status_code=503,
            detail="Models are nor available"
        )
    else:
        buffer_size = redis_client.llen(REDIS_LOG_KEY) or 0 # Verifier le contenu de la db 
        
        # Obtenir le nombre de prediction
        prediction_count = redis_client.llen("prediction_list") or 0
        
        return {
            "status": "operationnel",
            "models_loaded": isolation_forest_exist and one_class_svm_exist,
            "buffer_size": buffer_size,
            "threshold": THRESHOLD,
            "predictions_available": prediction_count
            
        }
        

@app.post("/log")
def add_log(log_entry:LogEntry):
    """
    add a sigle log entry to the redis buffer
    """
    # Transformation des logs au format json
    log_json = json.dumps(log_entry.dict())
    
    # Attribuer un id à un log
    
    log_id = redis_client.incr(REDIS_COUNTER_KEY)
    
    # Enregistrer dans redis
    redis_client.lpush(REDIS_LOG_KEY,log_json)
    
    # Connaitre la taille de notre buffer
    buffer_size = redis_client.llen(REDIS_LOG_KEY)
    
    if buffer_size > THRESHOLD:
        logs_json_list = redis_client.lrange(REDIS_LOG_KEY, 0, -1) # recuperer une liste de logs au format json
        
        logs_data = [json.loads(logs_json.decode('utf-8')) for logs_json in logs_json_list] # operation inverse du dump. Retourne une liste de dictionnaire
        
        redis_client.delete(REDIS_LOG_KEY) # Vider le buffer
        
        task = Process_batch_prediction.delay(json.dumps(logs_data))
        
        return {
            "message" : "logs add and batch processing data triggered",
            "task": task.id,
            "log_id": log_id
        }
        
    return {
        "message": "log add to buffer",
        "buffer_size": buffer_size,
        "log_id": log_id,
        "treshold": THRESHOLD
    }

@app.get("/prediction", response_model=Dict[str,Any])
def get_prediction_id():
    """Obtenir les identifiants de toutes les prédictions disponibles
    """
    # Obtenir la liste des ID de prédiction
    prediction_ids = redis_client.lrange("perdiction_list",0,-1)
    prediction_ids = [int(pid.decode('utf-8')) for pid in prediction_ids]
    
    return{
        "prediction_count": len(prediction_ids),
        "prediction_ids": prediction_ids
    }


    
# Recuperer les predictions lancée dans Process_batch_prediction
@app.get("/prediction/{prediction_id}", response_model=Dict[str,Any])
def get_prediction_by_id(prediction_id:int):
    """Get a specific predicition result by ID
    """
    # Get prediction from redis
    prediction_json = redis_client.get(f"{REDIS_PREFIX_PREDICTION_KEY}{prediction_id}") 
    
    if not prediction_json:
        raise HTTPException(status_code=404, detail="prediction not found")
    
    # Parse prediction data
    prediction_data = json.loads(prediction_json)
    
    return prediction_data
  
# Recuperer le status de la tâche
@app.get("/tasks/{task_id}")
def get_task_status(task_id:str):
    """Get status of Celery task
    """
    task = celery_app.AsyncResult(task_id) # Classe de celery permettant de récupérer le status d'une tâche à partir de son id
    
    response = {
        "task":task_id,
        "status":task.status
    }
    
    return response

# Route permettant de vider la db
@app.get("/clear_db")
def clear_redis_cache():
    redis_client.flushdb()
    return {
        "message": "db nettoyé"
    }
    
# Exécution du fichier

if __name__=="__main__":
    import uvicorn
    
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
