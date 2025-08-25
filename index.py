from flask import Flask, render_template, request, send_file
import pandas as pd
import os
from id3 import run_id3_analysis
from k_nn import run_knn_analysis

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
            return render_template("index.html", error="Por favor, sube un archivo CSV válido.")
        
        # Validar el número de pliegues
        try:
            cv_folds = int(validacion)
            if cv_folds < 2:
                return render_template("index.html", error="El número de pliegues debe ser al menos 2.")
        except ValueError:
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
                return render_template("index.html", error="Por favor, selecciona un algoritmo válido.")
            return render_template("index.html", results=results, filename=archivo.filename, algorithm_name=algorithm_name, cv_folds=cv_folds)
        except Exception as e:
            return render_template("index.html", error=str(e))
    
    return render_template("index.html")

@app.route("/uploads/<filename>")
def serve_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == "__main__":
    app.run(debug=True)