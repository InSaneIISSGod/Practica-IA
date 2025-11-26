# Caso Técnico – IA y Automatización
## 1. Objetivo
Implementar un flujo automatizado para clasificar tickets internos, generar un archivo JSON
estructurado y enviar los resultados a un API mock.
## 2. Requisitos técnicos
- Python 3.8+
- pandas
- scikit-learn
- nltk
- requests
- unidecode
- joblib
## 3. Ejecución
- Instalar dependencias: pip install -r requirements.txt
- Ejecutar: python procesar_tickets.py tickets.csv
## 4. Estructura del repositorio
```md
/src
├── api.py 
├── limpieza.py 
├── modelo.py 
├── procesar_tickets.py # Script principal de ejecución
├── tickets.csv # Dataset de entrada
│
└── /output 
    ├── resultado_jsonfinal.json
    ├── resultado_consumo_api.csv
    ├── metrics.txt
    ├── modelo_lr.joblib # Modelo entrenado 
    └── vectorizer.joblib # Vectorizador TF-IDF 
```
## 5. Decisiones técnicas
- Modelo ML: Regresión Logística, Es un modelo lineal robusto, rápido y eficiente para la clasificación de texto. Ofrece buena interpretabilidad y, con el ajuste de datos, se logró un Accuracy del 81%.
- Vectorización: TF-IDF con N-gramas (1, 2),	El uso de TF-IDF pondera la importancia de cada palabra. La adición de N-gramas (pares de palabras) permitió al modelo entender el contexto ("error de acceso" vs. solo "error"), mejorando la precisión.
- Priorización: La prioridad (Alta, Media, Baja) se define de forma explícita por reglas sencillas, asegurando que las palabras clave de alto impacto y facilite el entrenamiento del modelo.
- Unidecode: Se utiliza la libreria para facilitar la normalización del texto la tokenización para el entrenamiento.
- NLTK: Se realiza una verificación para asegurarse que la libreria cumpla con los recursos necesarios para ejecutar el script y evitar errores de dependencias. 
## 6. Mejoras propuestas
- Migrar a una base de datos NoSQL o tipo Json para facilitar la normalización de los datos.
- Extender el entrenamiento del modelo mediante tickets reales y extensos para que obtenga mayor accuracy.
- Probar otros algoritmos de lalibreria "scikit-learn" para comprobar si existen mejores alternativas y obtener mejores resultados.
- Implementar un "Cross-Validation" para detectar si el modelo esta overadjusted al dataset entrenado.
- Mejorar el procesamiento paralelo de API ya que actualmente al realizar el llamado desde una API fake es bastante lento.
- Analizar y mejorar las palabras clave del entrenamiento.