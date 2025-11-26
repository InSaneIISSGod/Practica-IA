import requests
import json
import pandas as pd

API_URL = "https://jsonplaceholder.typicode.com/posts"

def enviar_a_api(row):
    
    payload = {
        "id": int(row['ticket_id']), 
        "categoria": row['categoria_predicha'],
        "prioridad": row['prioridad'],
        "procesado_en": row['procesado_en']
    }
    
    try:
        # Intentar forzar la conversión a int si el ticket_id es un string de número
        if pd.notna(row['ticket_id']):
            payload['id'] = int(row['ticket_id'])
            
    except ValueError:
        pass 

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        
        if response.status_code == 201:
            return "POST Exitoso"
        else:
            return f"Error POST (Status: {response.status_code})"
            
    except requests.exceptions.RequestException as e:
        return f"Excepción de conexión: {e}"

def integrar_api(df_predicho):
    """Aplica la función de envío de API a todos los tickets procesados."""
    print("Tarea 4: Integrando con API mock...")

    df_api_data = df_predicho[['ticket_id', 'categoria_predicha', 'prioridad', 'procesado_en']]

    df_predicho['api_response'] = df_api_data.apply(enviar_a_api, axis=1)
    
    print("Tarea 4: Finalizada. Se realizaron los POST a la API.")
    return df_predicho