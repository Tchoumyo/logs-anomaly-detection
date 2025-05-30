import faker
from datetime import datetime, timedelta
import random
from constant import Constant

def generate_datetimes(start_date:datetime, end_date: datetime,number_logs:int) -> list[datetime]:
    """ 
        Génerer une liste de pas de temps selon une periode donnée
    """
    timestamps = []
    total_seconds = (end_date - start_date).total_seconds()
    for _ in range(number_logs):
        random_offset = random.random() * total_seconds
        timestamp = start_date + timedelta(seconds=random_offset)
        timestamps.append(timestamp)
    timestamps.sort()
    return timestamps

# Fonction permettant de generer les différents intervalles d'anomalies

def generate_anomaly_intervals(
        number_of_anomaly_intervals: int,
        number_of_logs: int,
        max_number_of_anomaly_per_interval: int,
        min_number_of_anomaly_per_interval: int,
        anomaly_types: list[str]
)->list[dict[str:int|str]]:
    """
        Génerer les intervales d'anomalies
    """
    
    intervals = []
    for _ in range(number_of_anomaly_intervals):
        start_idx = random.randint(0,number_of_logs - max_number_of_anomaly_per_interval) # calcul de l'indice de debut d'un interval d'anomalie
        nb_anomaly = random.randint(min_number_of_anomaly_per_interval, max_number_of_anomaly_per_interval) # Nombre d'anomaly par interval
        end_idx = min(start_idx + nb_anomaly, number_of_logs -1) # indice de fin
        
        intervals.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "type": random.choice(anomaly_types)
        })
    return intervals

# Definition de la methode principale chargée de génerer tout les logs de notre dataset

def generate_logs_dataset(
    start_date:datetime, 
    end_date: datetime, 
    number_of_anomaly_intervals: int,
    max_number_of_anomaly_per_interval: int,
    min_number_of_anomaly_per_interval: int,
    number_of_logs: int,
    http_error_list: list[str],
    logs_dataset_file_name: str,
    number_of_anomaly_ips: int,
    http_methods: list[str],
    http_normal_codes: list[str],
    http_error_codes : dict[str, list[str]],
    api_end_points: list[str]
):
    """
          Génerer le dataset des logs en incluant des anomalies
    """

    fak = faker.Faker()
    anomaly_ips = [fak.ipv4() for _ in range(number_of_anomaly_ips)]
    timestamps = generate_datetimes(start_date=start_date, end_date=end_date, number_logs=number_of_logs)

    anomaly_types = http_error_list.copy()
    anomaly_types.append("mixed_errors")

    anomaly_intervals = generate_anomaly_intervals(
        number_of_anomaly_intervals = number_of_anomaly_intervals,
        number_of_logs = number_of_logs,
        min_number_of_anomaly_per_interval = min_number_of_anomaly_per_interval,
        max_number_of_anomaly_per_interval = max_number_of_anomaly_per_interval,
        anomaly_types = anomaly_types
    )

    with open(logs_dataset_file_name, 'w') as file:
        file.write("timestamp,user_ip,method,status_code,end_point,response_time\n")
        
        for i in range(number_of_logs):
            timestamp=timestamps[i]
            user_ip = fak.ipv4()
            method = random.choice(http_methods)
            status_code = random.choice(http_normal_codes)
            end_point = random.choice(api_end_points)
            response_time = random.randint(10,300)

            for interval in anomaly_intervals:
                if interval["start_idx"] <= i <= interval["end_idx"]:
                    user_ip = random.choice(anomaly_ips)
                    if interval["type"] == "server_errors":
                        status_code = random.choice(http_error_codes["server_errors"])
                    elif interval["type"] == "client_errors":
                        status_code = random.choice(http_error_codes["client_errors"])
                    elif interval["type"] == "timeout_errors":
                        status_code = random.choice(http_error_codes["timeout_errors"])
                        response_time = random.randint(1000,5000)
                    else:
                        status_code = random.choice(list(http_error_codes.values()))[0]
                        response_time = random.randint(1000,5000)
            file.write(f"{timestamp},{user_ip},{method},{status_code},{end_point},{response_time}\n")

    print(f"le fichier csv '{logs_dataset_file_name}' a été généré avec  {number_of_logs}, logs et {number_of_anomaly_intervals} plages d'anomalies")


if __name__=="__main__":
    generate_logs_dataset(
        start_date = Constant.LOGS_START_DATE,
        end_date = Constant.LOGS_END_DATE,
        number_of_anomaly_intervals = Constant.NUMBERS_OF_ANOMALY_INTERVALS,
        number_of_logs=Constant.NUMBER_LOGS,
        min_number_of_anomaly_per_interval=Constant.MIN_NUMBER_OF_ANOMALY_PER_INTERVAL,
        max_number_of_anomaly_per_interval=Constant.MAX_NUMBER_OF_ANOMALY_PER_INTERVAL,
        http_error_list=list(Constant.HTTP_ERRORS_CODES.keys()),
        number_of_anomaly_ips=Constant.NUMBER_OF_ANOMALY_IPS,
        logs_dataset_file_name=Constant.LOGS_DATASET_FILE_NAME,
        http_methods=Constant.HTTP_METHODS,
        http_normal_codes=Constant.HTTP_NORMAL_CODES,
        http_error_codes=Constant.HTTP_ERRORS_CODES,
        api_end_points=Constant.API_ENDPOINTS
    )