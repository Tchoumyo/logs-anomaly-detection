import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from constant import Constant

class LogPreprocessor:
    def __init__(self):
        self.preprocessor = None # Contiendra le pipeline de pretraitement de nos caractéristiques

    def Parse_features(self, df) -> pd.DataFrame:
        df['status_code'] = df['status_code'].astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['timestamp'].isin([5,6]).astype(int)

        user_stats = df.groupby('user_ip').agg(
                
                error_count = ("status_code", lambda x: (x.str.startswith("5") | x.str.startswith("4")).sum()),
                unique_error_type = ("status_code", lambda x: x[x.str.startswith("5") | x.str.startswith("4")].nunique()),
                avg_response_time = ("response_time", 'mean'),
                max_response_time = ("response_time", 'max')
        ).reset_index()

        df = df.merge(user_stats, how='left', on='user_ip')

        df["response_time_category"] = pd.cut(
            df['response_time'], bins=[-1,100,200,300],
            labels=['fast','normal','slow']
        )

        df["is_potential_anomalous"] = (
            (df['response_time'] > 200) | ((df['status_code'].str.startswith("5")) | (df['status_code'].str.startswith("4")))
        ).astype(int)

        return df
    

    def build_preprocessor(self,df):
        numeric_features = ["response_time","hour_of_day","day_of_week","month","error_count","unique_error_type","avg_response_time","max_response_time"]
        categorical_features = ["is_weekend","method","end_point","is_potential_anomalous","response_time_category"]

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num',numeric_transformer,numeric_features),
                ('cat', categorical_transformer,categorical_features)
            ]
        )
        
        return self
    
    def fit_transform(self,df):
        df_parsed = self.Parse_features(df)
        self.build_preprocessor(df_parsed)

        X = df_parsed.drop(['timestamp','status_code','user_ip'], axis=1)

        X_preprocessed = self.preprocessor.fit_transform(X)

        return X_preprocessed, X
    

    # Séparation de nos données en données d'entrainement et de test
    def split_dataset(self,df:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
        """
        Séparer notre dataset en deux : trainset et testset
        """
        df = df.sample(n=len(df)) # Mélanger la totalité de nos données
        train_set_size = int(Constant.TRAIN_SET_RATIO * len(df))
        df_train = df.iloc[:train_set_size]
        df_test = df.iloc[train_set_size:]
        return df_train,df_test


if __name__=="__main__" :
    df  = pd.read_csv('data/logs_dataset.csv')
    LogPreprocessor = LogPreprocessor()
    X_preprocessed,X = LogPreprocessor.fit_transform(df)
    print('operation réussi')
    print('X_preprocessed dimension', X_preprocessed.shape)
    print('X dimension', X.shape)
    print('X_preprocessed', X_preprocessed)
    print('X', X.head(10))
    X.to_csv('data/logs_preprocessed.csv')