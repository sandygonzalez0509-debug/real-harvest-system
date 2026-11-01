
import os
import openai
from pinecone import Pinecone
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Inicialización del motor analítico bajo el Protocolo V10.12
openai.api_key = os.environ.get("OPENAI_API_KEY")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("sports-analytics-v12")

class SportsAnalyticsHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_data = json.loads(post_data.decode('utf-8'))
        
        game_id = request_data.get("game_id")
        odds_movement = request_data.get("odds_movement")
        injury_report = request_data.get("injury_report")
        
        # Consulta semántica en la base de datos vectorial con la literatura experta
        vector_query = f"Análisis predictivo para juego {game_id} con movimiento de cuotas {odds_movement}"
        embedding_response = openai.embeddings.create(
            input=[vector_query],
            model="text-embedding-3-small"
        )
        query_vector = embedding_response.data[0].embedding
        
        search_results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        
        # Evaluación de valor (+EV) respaldada por los modelos teóricos
        prompt_content = f"""
        Actúa como el Director del Ecosistema de Trading Deportivo bajo el Protocolo V10.12.
        Contexto de literatura experta recuperada: {search_results}
        Datos actuales del mercado:
        - Movimiento de cuotas: {odds_movement}
        - Reporte de lesiones y carga: {injury_report}
        
        Evalúa estrictamente si existe una ventaja matemática real sobre la casa. Si no hay valor claro, responde obligatoriamente: 'No hay valor, protegemos el bankroll'. Si existe margen positivo, detalla la ejecución exacta.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_content}],
            temperature=0.1
        )
        
        analysis_result = response.choices[0].message.content
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_payload = {"status": "success", "analysis": analysis_result}
        self.wfile.write(json.dumps(response_payload).encode('utf-8'))
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Real Harvest System V10.12 - Operativo")

def run(server_class=HTTPServer, handler_class=SportsAnalyticsHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Motor analítico V10.12 activo en el puerto {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss

# 1. CONSTRUCCIÓN DEL DATAFRAME Y VARIABLES (FEATURE ENGINEERING)
def estructurar_features(df):
    """
    Recibe los datos crudos y calcula las variables de presión, 
    sesgo arbitral e incentivos comerciales.
    """
    # Sesgo y Presión de Masas (Soccernomics & Favoritism Under Social Pressure)
    df['attendance_ratio'] = df['asistencia_real'] / df['capacidad_estadio']
    
    # Incentivos Contractuales (Playing for Keeps)
    # 1 si es su último año de contrato, 0 si no lo es
    df['contract_year_flag'] = np.where(df['dias_restantes_contrato'] <= 365, 1, 0)
    
    # Presión Institucional en Playoffs (Subperfect Game)
    # Ejemplo: Si el equipo local va perdiendo la serie, el sistema asigna peso
    df['extends_series_incentive'] = np.where(
        (df['juego_de_playoffs'] == 1) & (df['equipo_local_abajo_en_serie'] == 1), 1, 0
    )
    
    # Microestructura del Mercado (Levitt & Paul/Weinbach)
    # Diferencia entre el % de dinero (handle) y el % de tickets (público)
    df['sharp_money_divergence'] = df['money_pct'] - df['ticket_pct']
    
    # Public Shade: Diferencia entre la cuota de la casa y tu modelo base
    df['public_shade'] = df['cuota_implícita_casa'] - df['probabilidad_modelo_base']
    
    return df

# 2. ENTRENAMIENTO Y CALIBRACIÓN DEL MODELO
def entrenar_modelo_ev(df, features, target):
    """
    Entrena una regresión logística para calibrar la probabilidad real,
    eliminando las variables que no superan la prueba empírica.
    """
    X = df[features]
    y = df[target] # 1 si el evento/prop ocurrió, 0 si falló
    
    # Separación para validar fuera de la muestra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
    modelo.fit(X_train, y_train)
    
    # Predicción probabilística calibrada
    probabilidades = modelo.predict_proba(X_test)[:, 1]
    
    # Evaluación rigurosa (Superforecasting)
    brier = brier_score_loss(y_test, probabilidades)
    print(f"Brier Score (Calibración): {brier:.4f} (Más cerca a 0 es mejor)")
    
    return modelo, X_test, probabilidades

# 3. VALIDACIÓN DEFINITIVA: CLOSING LINE VALUE (CLV)
def calcular_clv_y_ev(cuota_tomada, cuota_cierre, prob_modelo):
    """
    Calcula el CLV para medir la eficiencia del sistema y el EV 
    para proteger el bankroll antes de ejecutar.
    """
    # CLV: Si tomaste +150 (2.50) y cierra en +120 (2.20), superaste al mercado
    clv_porcentaje = (cuota_tomada / cuota_cierre) - 1
    
    # Valor Esperado (EV) matemático
    ev_matematico = (prob_modelo * cuota_tomada) - 1
    
    return clv_porcentaje, ev_matematico

