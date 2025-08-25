import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_id3_analysis(filepath, cv_folds=10):
    # Cargar el CSV
    try:
        df = pd.read_csv(filepath)
        if 'species' not in df.columns:
            raise ValueError("El archivo CSV debe contener una columna 'species'.")
        X = df.drop(columns=["species"])
        y = df["species"]
    except Exception as e:
        raise ValueError(f"Error al cargar el CSV: {str(e)}")

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear y entrenar el modelo ID3
    id3 = DecisionTreeClassifier(criterion="entropy", random_state=42)
    id3.fit(X_train, y_train)

    # Evaluación del modelo
    y_pred = id3.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Guardar matriz de confusión como imagen
    confusion_matrix_img = f"confusion_matrix_{os.path.basename(filepath)}.png"
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                xticklabels=id3.classes_, yticklabels=id3.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Matriz de Confusión ID3")
    plt.tight_layout()
    plt.savefig(os.path.join('Uploads', confusion_matrix_img))
    plt.close()

    # Guardar visualización del árbol como imagen
    tree_img = f"decision_tree_{os.path.basename(filepath)}.png"
    plt.figure(figsize=(14, 8))
    plot_tree(id3, 
              feature_names=X.columns, 
              class_names=id3.classes_,
              filled=True, rounded=True, fontsize=10)
    plt.title("Árbol de Decisión (ID3)")
    plt.tight_layout()
    plt.savefig(os.path.join('Uploads', tree_img))
    plt.close()

    # Validación cruzada
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(id3, X, y, cv=cv, scoring="accuracy")

    # Guardar el modelo
    model_filename = f"modelo_id3_{os.path.basename(filepath)}.joblib"
    joblib.dump(id3, os.path.join('Uploads', model_filename))

    # Probar con un nuevo ejemplo (usar valores promedio de las características)
    example_input = np.array([X.mean().values]).reshape(1, -1)
    modelo_cargado = joblib.load(os.path.join('Uploads', model_filename))
    prediccion = modelo_cargado.predict(example_input)[0]

    # Retornar resultados
    return {
        'accuracy': round(accuracy, 4),
        'classification_report': class_report,
        'confusion_matrix_img': confusion_matrix_img,
        'tree_img': tree_img,
        'cv_scores': [round(score, 4) for score in scores],
        'cv_mean': round(scores.mean(), 4),
        'cv_std': round(scores.std(), 4),
        'example_input': example_input[0].tolist(),
        'example_prediction': prediccion
    }