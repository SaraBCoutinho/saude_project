import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier # Exemplo de estimador
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

df = pd.read_csv('data_processed.csv')
target = 'Risco_Doenca' 
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(X_train.drop(columns=['ID'])) # Remove ID apenas para o cálculo

# Filtrando o treino
X_train_cleaned = X_train[outliers == 1]
y_train_cleaned = y_train[outliers == 1]

# --- 3. Definição do Pipeline de Pré-processamento ---
# Aqui removemos o ID definitivamente e aplicamos Imputer + Scaler + PCA
features = [col for col in X_train.columns if col != 'ID']

# 1. Definição dos seus componentes de pré-processamento
preprocessor_pca = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])

preprocessor_simple = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

# 2. Criamos um dicionário com os diferentes pipelines que queremos testar
cenarios = {
    "RF_com_PCA": Pipeline(steps=[
        ('preprocessor', preprocessor_pca),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ]),
    "RF_sem_PCA": Pipeline(steps=[
        ('preprocessor', preprocessor_simple),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
}

# 3. Execução Iterativa no MLflow
mlflow.set_experiment("Projeto_Saude")

for nome_run, pipeline_objeto in cenarios.items():
    with mlflow.start_run(run_name=nome_run):
        
        # Treino
        pipeline_objeto.fit(X_train_cleaned[features], y_train_cleaned)
        
        # Predição
        y_pred = pipeline_objeto.predict(X_test[features])
        
        # Métricas
        metrics = {
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Logs automáticos
        mlflow.log_param("pca_active", "Sim" if "PCA" in nome_run else "Não")
        mlflow.log_metrics(metrics)
        
        # Log do modelo específico desta rodada
        mlflow.sklearn.log_model(pipeline_objeto, "model")
        
        print(f"Finalizado: {nome_run}")


# Supondo que o RF_sem_PCA foi o melhor:
melhor_pipeline = cenarios["RF_sem_PCA"]

# Salva o arquivo pronto para o Streamlit
joblib.dump(melhor_pipeline, "modelo_final_saude.pkl")
print("Modelo exportado com sucesso como 'modelo_final_saude.pkl'")