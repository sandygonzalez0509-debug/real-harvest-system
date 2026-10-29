
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

def run(server_class=HTTPServer, handler_class=SportsAnalyticsHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Motor analítico V10.12 activo en el puerto {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
