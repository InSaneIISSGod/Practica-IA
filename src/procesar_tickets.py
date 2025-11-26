import sys
import pandas as pd
import json
import os
import datetime

from limpieza import procesar_dataset
from modelo import entrenar_modelo, predecir_y_priorizar
from api import integrar_api

# Rutas de salida
OUTPUT_DIR = 'output'
OUTPUT_JSON_FILE = os.path.join(OUTPUT_DIR, 'resultado_jsonfinal.json')
METRICS_FILE = os.path.join(OUTPUT_DIR, 'metrics.txt')
API_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'resultado_consumo_api.csv')

def generar_json(df):
    
    json_data = df[['ticket_id', 'categoria_predicha', 'prioridad', 'procesado_en']].copy()
    json_data['ticket_id'] = json_data['ticket_id'].astype(str)
    
    json_list = json_data.to_dict('records')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, indent=4, ensure_ascii=False)
        
    print(f"Tarea 3: Archivo de salida JSON generado en '{OUTPUT_JSON_FILE}'.")

def guardar_metricas(accuracy, cm):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        f.write("--- Métricas del Modelo de Clasificación ---\n")
        f.write(f"Fecha de ejecución: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Matriz de Confusión:\n")
        f.write(cm.to_string())
    print(f"Tarea 6: Métricas guardadas en '{METRICS_FILE}'.")

def guardar_resultados_api(df):
    df[['ticket_id', 'categoria_predicha', 'prioridad', 'procesado_en', 'api_response']].to_csv(
        API_RESULTS_FILE, index=False
    )
    print(f"Tarea 6: Resultado de consumo de API guardado en '{API_RESULTS_FILE}'.")

def flujo_principal(csv_file):
    
    if not os.path.exists(csv_file):
        print(f"Error: No se encontró el archivo {csv_file}")
        sys.exit(1)

    print(f"*** Iniciando el procesamiento de tickets desde '{csv_file}' ***")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        sys.exit(1)
        
    df.columns = df.columns.str.lower().str.strip()
    
    if 'id' in df.columns and 'ticket_id' not in df.columns:
        df.rename(columns={'id': 'ticket_id'}, inplace=True)
    elif 'ticket_id' not in df.columns:
        df.insert(0, 'ticket_id', range(1, 1 + len(df)))
        
    # Tarea 1: Procesamiento del dataset
    df_procesado = procesar_dataset(df.copy())
    
    # Tarea 2: Entrenamiento y Predicción CONDICIONAL
    if 'categoria' in df_procesado.columns:
        print("\n[INFO] Columna 'categoria' encontrada. Ejecutando FASE DE ENTRENAMIENTO.")
        
        # Tarea 2: Entrenamiento del modelo
        _, _, accuracy, cm = entrenar_modelo(df_procesado.copy())
        
        guardar_metricas(accuracy, pd.DataFrame(cm))
        
        # Tarea 2/3: Predicción y Priorización
        df_prediccion = predecir_y_priorizar(df_procesado.copy())
        
    else:
        print("\n[ATENCIÓN] Columna 'categoria' NO encontrada. Iniciando modo de Predicción.")
        
        # Tarea 2/3: Solo Predicción y Priorización
        try:
             df_prediccion = predecir_y_priorizar(df_procesado.copy())
        except FileNotFoundError:
             print("\nERROR: El modelo entrenado ('modelo_lr.joblib') no existe en 'output/'.")
             print("Debe ejecutar el script con un archivo que SÍ contenga la columna 'categoria' para entrenar el modelo primero.")
             sys.exit(1)
             
    # Tarea 3: Generación de salida automática (.json)
    generar_json(df_prediccion)
    
    # Tarea 4: Integración con API mock
    df_final = integrar_api(df_prediccion)
    
    # Guarda el resultado del consumo de API 
    guardar_resultados_api(df_final)
    
    print("\n*** Flujo end-to-end completado exitosamente. ***")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python procesar_tickets.py <archivo_csv>")
        sys.exit(1)
        
    csv_input = sys.argv[1]
    flujo_principal(csv_input)