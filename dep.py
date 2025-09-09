import pandas as pd
import joblib
import logging
import sys
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Cargar el modelo
    model_path = "uploads/modelo_id3_student-mat.csv.joblib"
    logger.debug(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Cargar el dataset de entrenamiento para obtener columnas
    train_df = pd.read_csv("Uploads/student-mat.csv", sep=";")
    logger.debug(f"Loaded training dataset with shape: {train_df.shape}")
    train_df["target"] = np.where(train_df["G3"] >= 10, "pass", "fail")
    X_train = train_df.drop(columns=["G3", "target"])
    X_train = pd.get_dummies(X_train)
    logger.debug(f"Training columns after one-hot encoding: {X_train.columns.tolist()}")

    # Cargar el CSV de predicción
    pred_df = pd.read_csv("uploads/prediction_data.csv", sep=";")
    logger.debug(f"Loaded prediction dataset with shape: {pred_df.shape}")
    logger.debug(f"Prediction columns: {pred_df.columns.tolist()}")

    # Asegurarse de que no haya columna G3 en el CSV de predicción
    if 'G3' in pred_df.columns:
        pred_df = pred_df.drop(columns=['G3'])
        logger.debug("Dropped G3 column from prediction data")

    # Aplicar codificación one-hot al CSV de predicción
    pred_df = pd.get_dummies(pred_df)
    logger.debug(f"Prediction columns after one-hot encoding: {pred_df.columns.tolist()}")

    # Alinear columnas con el dataset de entrenamiento
    pred_df = pred_df.reindex(columns=X_train.columns, fill_value=0)
    logger.debug(f"Prediction data shape after reindex: {pred_df.shape}")

    # Hacer predicciones
    predictions = model.predict(pred_df.values)
    logger.debug(f"Predictions: {predictions}")

    # Imprimir resultados
    print("Predicciones:", predictions)

except Exception as e:
    logger.error(f"Error during execution: {str(e)}", exc_info=True)
    sys.exit(1)