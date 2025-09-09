from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import os
import logging
import joblib

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from id3 import run_id3_analysis
    from k_nn import run_knn_analysis
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    raise

app = Flask(__name__)

# Directorio para guardar archivos temporales
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def check_model_exists(filename, algorithm):
    """Verifica si existe un modelo guardado para el archivo y algoritmo."""
    model_filename = f"modelo_{algorithm.lower()}_{filename}.joblib"
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
    return os.path.exists(model_path)

@app.route("/", methods=["GET", "POST"])
def home():
    model_exists = False
    filename = None
    algorithm = None
    if request.method == "POST":
        archivo = request.files.get("archivo")
        algorithm = request.form.get("opcion")
        validacion = request.form.get("validacion")
        
        if not archivo or not archivo.filename.endswith('.csv'):
            logger.error("Archivo no válido o no es CSV")
            return render_template("index.html", error="Por favor, sube un archivo CSV válido.")
        
        try:
            cv_folds = int(validacion)
            if cv_folds < 2:
                logger.error("Número de pliegues menor a 2")
                return render_template("index.html", error="El número de pliegues debe ser al menos 2.")
        except ValueError:
            logger.error("Valor de validación cruzada no es entero")
            return render_template("index.html", error="El valor de validación cruzada debe ser un número entero.")

        filename = archivo.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
        archivo.save(filepath)
        
        try:
            if algorithm == "id3":
                results = run_id3_analysis(filepath, cv_folds=cv_folds)
                algorithm_name = "ID3"
            elif algorithm == "knn":
                results = run_knn_analysis(filepath, cv_folds=cv_folds)
                algorithm_name = "K-NN"
            else:
                logger.error("Algoritmo no válido seleccionado")
                return render_template("index.html", error="Por favor, selecciona un algoritmo válido.")
            model_exists = check_model_exists(filename, algorithm)
            logger.debug(f"Rendering index.html with filename={filename}, algorithm_name={algorithm_name}, model_exists={model_exists}")
            return render_template("index.html", results=results, filename=filename, algorithm_name=algorithm_name, cv_folds=cv_folds, model_exists=model_exists, algorithm=algorithm)
        except Exception as e:
            logger.error(f"Error en análisis: {str(e)}")
            return render_template("index.html", error=str(e))
    
    return render_template("index.html", model_exists=model_exists, filename=filename, algorithm=algorithm)

@app.route("/get_columns", methods=["POST"])
def get_columns():
    archivo = request.files.get("archivo") or request.files.get("new_csv")
    if not archivo or not archivo.filename.endswith('.csv'):
        logger.error("Archivo no válido o no es CSV en /get_columns")
        return jsonify({'error': 'Por favor, sube un archivo CSV válido.'}), 400
    
    try:
        archivo.seek(0)
        df = pd.read_csv(archivo, sep=';', encoding='utf-8')
        columns = df.columns.tolist()
        logger.debug(f"Columnas extraídas: {columns}")
        return jsonify({'columns': columns})
    except Exception as e:
        logger.error(f"Error al leer CSV en /get_columns: {str(e)}")
        return jsonify({'error': f'Error al leer el CSV: {str(e)}'}), 400

@app.route("/predict", methods=["GET", "POST"])
def predict():
    filename = request.args.get('filename')
    algorithm = request.args.get('algorithm')
    logger.debug(f"Predict route called with filename={filename}, algorithm={algorithm}")
    if not filename or not algorithm:
        logger.error("Missing filename or algorithm in predict route")
        return render_template("index.html", error="Missing filename or algorithm", filename=filename, algorithm_name=algorithm)
    
    try:
        new_filepath = None
        columns = None
        if request.method == "POST" and 'new_csv' in request.files:
            new_csv = request.files['new_csv']
            if new_csv and new_csv.filename.endswith('.csv'):
                new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_csv.filename)
                new_csv.save(new_filepath)
                logger.debug(f"New CSV uploaded: {new_filepath}")
                df = pd.read_csv(new_filepath, sep=';', encoding='utf-8')
                if 'G3' in df.columns:
                    logger.warning("New CSV contains G3, it will be ignored")
                columns = df.drop(columns=['G3'] if 'G3' in df.columns else []).columns.tolist()
                filename = new_csv.filename
            else:
                logger.error("Invalid new CSV file")
                return render_template("predict.html", error="Por favor, sube un archivo CSV válido.", filename=filename, algorithm_name=algorithm)
        else:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.debug(f"Loading original file: {filepath}")
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                raise FileNotFoundError(f"File {filename} not found")
            df = pd.read_csv(filepath, sep=';', encoding='utf-8')
            if 'G3' not in df.columns:
                logger.error("G3 column missing in original CSV")
                raise ValueError("El archivo CSV debe contener una columna 'G3'.")
            columns = df.drop(columns=["G3"]).columns.tolist()
        
        logger.debug(f"Columns extracted: {columns}")
        return render_template("predict.html", columns=columns, filename=filename, algorithm_name=algorithm, new_filepath=new_filepath)
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return render_template("predict.html", error=str(e), filename=filename, algorithm_name=algorithm)

@app.route("/make_prediction", methods=["POST"])
def make_prediction():
    filename = request.form.get('filename')
    algorithm = request.form.get('algorithm')
    logger.debug(f"Received make_prediction request: filename={filename}, algorithm={algorithm}")
    try:
        # Cargar el modelo
        model_filename = f"modelo_{algorithm.lower()}_{filename}.joblib"
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
        logger.debug(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # Verificar si hay un nuevo CSV para predicción
        new_csv = request.files.get('new_csv')
        if new_csv and new_csv.filename.endswith('.csv'):
            # Guardar el nuevo CSV
            new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_csv.filename)
            new_csv.save(new_filepath)
            logger.debug(f"New CSV saved: {new_filepath}")
            
            # Leer el nuevo CSV
            df = pd.read_csv(new_filepath, sep=';', encoding='utf-8')
            if 'G3' in df.columns:
                df = df.drop(columns=['G3'])
            
            # Preparar datos
            df = pd.get_dummies(df)
            
            # Alinear con las columnas del modelo
            original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            train_df = pd.read_csv(original_filepath, sep=';', encoding='utf-8')
            X_train = train_df.drop(columns=["G3"])
            X_train = pd.get_dummies(X_train)
            df = df.reindex(columns=X_train.columns, fill_value=0)
            
            # Realizar predicciones
            predictions = model.predict(df.values)
            logger.debug(f"Predictions: {predictions}")
            return jsonify({'predictions': predictions.tolist()})
        else:
            return jsonify({'error': 'Debe subir un CSV para predicción'}), 400
    except Exception as e:
        logger.error(f"Error in make_prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route("/uploads/<filename>")
def serve_image(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.debug(f"Serving file from: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404
        return send_file(file_path)
    except Exception as e:
        logger.error(f"Error al servir archivo {filename}: {str(e)}")
        return jsonify({'error': f'Error al servir el archivo: {str(e)}'}), 404

if __name__ == "__main__":
    app.run(debug=True)