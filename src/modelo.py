import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import datetime
from unidecode import unidecode

# Constantes
CATEGORIAS = ['TI', 'RRHH', 'Finanzas', 'Soporte General']
MODELO_PATH = 'output/modelo_lr.joblib'
VECTORIZER_PATH = 'output/vectorizer.joblib'

def calcular_prioridad(descripcion):

    if pd.isna(descripcion):
        return 'baja'
        
    descripcion = unidecode(str(descripcion).lower())
    
    palabras_clave_alta = ['caido', 'urgente', 'bloqueado', 'no funciona', 'critico', 'no arranca']
    palabras_clave_media = ['error', 'modificar', 'acceso', 'problema', 'lentitud', 'no conecta']

    if any(keyword in descripcion for keyword in palabras_clave_alta):
        return 'alta'
    
    if any(keyword in descripcion for keyword in palabras_clave_media):
        return 'media'
        
    if len(descripcion.split()) > 30: 
        return 'media'
        
    return 'baja'

def entrenar_modelo(df):
    print(" Tarea 2: Entrenando el modelo de clasificación (Logistic Regression)...")
    
    X = df['descripcion_limpia']
    y = df['categoria']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=500, random_state=42, C=0.2)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=CATEGORIAS)
    
    print("\n--- Métricas del Modelo ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Matriz de Confusión:\n", pd.DataFrame(cm, index=CATEGORIAS, columns=CATEGORIAS))
    print("---------------------------\n")

    # Guardar modelo y vectorizador
    joblib.dump(model, MODELO_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"-> Artefactos de ML guardados en 'output/'.")
    
    return model, vectorizer, accuracy, cm

def predecir_y_priorizar(df):
    print("Tarea 3: Realizando predicciones y determinando prioridad...")
    
    try:
        model = joblib.load(MODELO_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        # Lanzar la excepción de nuevo para que sea capturada en procesar_tickets.py
        raise FileNotFoundError("Error: El modelo o vectorizador no existe. Ejecuta primero la fase de entrenamiento.")

    # 1. Predecir categorías (Tarea 2)
    X_clean = df['descripcion_limpia']
    X_vec = vectorizer.transform(X_clean)
    df['categoria_predicha'] = model.predict(X_vec)

    # 2. Determinar Prioridad (Tarea 3)
    df['prioridad'] = df['descripcion'].apply(calcular_prioridad)
    
    # 3. Generar columna de fecha de procesamiento (Tarea 3)
    now_iso = datetime.datetime.now().isoformat(timespec='seconds')
    df['procesado_en'] = now_iso

    print("Tarea 3: Predicciones y prioridades calculadas.")
    return df