import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_knn_analysis(filepath, cv_folds=10):
    # Cargar el CSV
    try:
        df = pd.read_csv(filepath, sep=';')
        if 'G3' not in df.columns:
            raise ValueError("El archivo CSV debe contener una columna 'G3'.")
        df['target'] = np.where(df['G3'] >= 10, 'pass', 'fail')
        X = df.drop(columns=["G3", "target"])
        X = pd.get_dummies(X)
        y = df["target"]
    except Exception as e:
        raise ValueError(f"Error al cargar el CSV: {str(e)}")

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear y entrenar el modelo KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Evaluación del modelo
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Guardar matriz de confusión como imagen
    confusion_matrix_img = f"confusion_matrix_knn_{os.path.basename(filepath)}.png"
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                xticklabels=knn.classes_, yticklabels=knn.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Matriz de Confusión K-NN")
    plt.tight_layout()
    plt.savefig(os.path.join('uploads', confusion_matrix_img))
    plt.close()

    # Guardar visualización del espacio 2D (studTime vs absences)
    knn_img = f"knn_plot_{os.path.basename(filepath)}.png"
    feature_x = "studytime"
    feature_y = "absences"
    X_vis = df[[feature_x, feature_y]].values
    y_vis = df["target"].values
    knn_vis = KNeighborsClassifier(n_neighbors=5)
    knn_vis.fit(X_vis, y_vis)
    nuevo_vis = np.array([[2, 5]])  # Ejemplo con 2 características
    distancias, indices = knn_vis.kneighbors(nuevo_vis)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=feature_x, y=feature_y, hue="target", data=df, palette="Set2", s=60)
    plt.scatter(nuevo_vis[0][0], nuevo_vis[0][1], color='red', label='Nuevo ejemplo', s=100, edgecolor='black', marker='X')
    for idx in indices[0]:
        vecino = X_vis[idx]
        plt.plot([nuevo_vis[0][0], vecino[0]], [nuevo_vis[0][1], vecino[1]], 'k--', linewidth=0.8)
    plt.title("K-NN en espacio 2D (studytime vs absences)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('uploads', knn_img))
    plt.close()

    # Validación cruzada
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")

    # Guardar el modelo
    model_filename = f"modelo_knn_{os.path.basename(filepath)}.joblib"
    joblib.dump(knn, os.path.join('uploads', model_filename))

    # Probar con un nuevo ejemplo (usar valores promedio de las características)
    example_input = np.array([X.mean().values]).reshape(1, -1)
    modelo_cargado = joblib.load(os.path.join('uploads', model_filename))
    prediccion = modelo_cargado.predict(example_input)[0]

    # Retornar resultados
    return {
        'accuracy': round(accuracy, 4),
        'classification_report': class_report,
        'confusion_matrix_img': confusion_matrix_img,
        'knn_img': knn_img,
        'cv_scores': [round(score, 4) for score in scores],
        'cv_mean': round(scores.mean(), 4),
        'cv_std': round(scores.std(), 4),
        'example_input': example_input[0].tolist(),
        'example_prediction': prediccion
    }