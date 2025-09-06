from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import os
import logging

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

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        archivo = request.files.get("archivo")
        algorithm = request.form.get("opcion")
        validacion = request.form.get("validacion")
        
        # Validar el archivo
        if not archivo or not archivo.filename.endswith('.csv'):
            logger.error("Archivo no válido o no es CSV")
            return render_template("index.html", error="Por favor, sube un archivo CSV válido.")
        
        # Validar el número de pliegues
        try:
            cv_folds = int(validacion)
            if cv_folds < 2:
                logger.error("Número de pliegues menor a 2")
                return render_template("index.html", error="El número de pliegues debe ser al menos 2.")
        except ValueError:
            logger.error("Valor de validación cruzada no es entero")
            return render_template("index.html", error="El valor de validación cruzada debe ser un número entero.")

        # Guardar el archivo subido
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
        archivo.save(filepath)
        
        # Ejecutar análisis según el algoritmo seleccionado
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
            return render_template("index.html", results=results, filename=archivo.filename, algorithm_name=algorithm_name, cv_folds=cv_folds)
        except Exception as e:
            logger.error(f"Error en análisis: {str(e)}")
            return render_template("index.html", error=str(e))
    
    return render_template("index.html")

@app.route("/get_columns", methods=["POST"])
def get_columns():
    archivo = request.files.get("archivo")
    if not archivo or not archivo.filename.endswith('.csv'):
        logger.error("Archivo no válido o no es CSV en /get_columns")
        return jsonify({'error': 'Por favor, sube un archivo CSV válido.'}), 400
    
    try:
        # Resetear la posición del archivo
        archivo.seek(0)
        df = pd.read_csv(archivo, sep=';', encoding='utf-8')
        columns = df.columns.tolist()
        logger.debug(f"Columnas extraídas: {columns}")
        return jsonify({'columns': columns})
    except Exception as e:
        logger.error(f"Error al leer CSV en /get_columns: {str(e)}")
        return jsonify({'error': f'Error al leer el CSV: {str(e)}'}), 400

@app.route("/uploads/<filename>")
def serve_image(filename):
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except Exception as e:
        logger.error(f"Error al servir archivo {filename}: {str(e)}")
        return jsonify({'error': f'Error al servir el archivo: {str(e)}'}), 404

if __name__ == "__main__":
    app.run(debug=True)