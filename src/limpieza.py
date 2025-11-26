import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import nltk
import sys 

# Bloque de descarga automática de NLTK
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab/spanish/')
    print("[NLTK] Recursos cargados correctamente.")

except LookupError:
    #Si un LookupError ocurre, procedemos a descargar
    print("\n[NLTK] Recursos no encontrados. Intentando descarga automática...")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True) 
        
        print("[NLTK] Descarga completada. El flujo continuará.")
    except Exception as e:
        print(f"\n[ERROR CRÍTICO NLTK] No se pudo completar la descarga automática. Mensaje: {e}")
        sys.exit(1)
        
# Lista de stopwords en español
STOP_WORDS_ES = set(stopwords.words('spanish'))

def normalizar_texto(texto):

    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = unidecode(texto)
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return texto

def eliminar_stopwords(texto):
    if not texto:
        return ""
    
    # Especificamos "español"
    tokens = word_tokenize(texto, language='spanish') 
    
    tokens_limpios = [word for word in tokens if word not in STOP_WORDS_ES]
    return " ".join(tokens_limpios)

def procesar_dataset(df):

    print("Tarea 1: Procesando el dataset...")
    
    # Manejar el caso donde 'descripcion' no existe
    if 'descripcion' not in df.columns:
        print("[ERROR] El archivo de entrada no contiene la columna 'descripcion'.")
        sys.exit(1)
        
    df['descripcion_normalizada'] = df['descripcion'].apply(normalizar_texto)
    df['descripcion_limpia'] = df['descripcion_normalizada'].apply(eliminar_stopwords)
    print("Tarea 1: Finalizada. Columna 'descripcion_limpia' creada.")
    return df