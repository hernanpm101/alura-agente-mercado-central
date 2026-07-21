"""
Alura Agente — Mercado Central 24h
Versión de deploy: usa TF-IDF (scikit-learn) para la búsqueda de fragmentos relevantes,
en vez de un modelo de embeddings neuronal. Esto elimina por completo el problema de
memoria que sufríamos con modelos de IA para embeddings en el tier gratuito de la nube
(512 MB de RAM no alcanzan para cargar un modelo neuronal + el resto del stack).

TF-IDF es una técnica clásica de recuperación de información basada en palabras clave
(no en significado semántico) — el consumo de RAM es de apenas unos pocos MB, sin
depender de ningún modelo externo. El notebook de Colab sigue usando el enfoque
semántico con embeddings de IA; este cambio aplica solo a la versión desplegada,
donde el límite de memoria es la restricción real del entorno gratuito.

Expone:
  GET  /             -> formulario simple para preguntarle al agente
  POST /preguntar     -> endpoint JSON: {"pregunta": "..."} -> {"respuesta": "...", "fuentes": [...]}
"""

import os
import json
from flask import Flask, request, jsonify, render_template_string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. Cargar los fragmentos precalculados y armar el índice TF-IDF
# ---------------------------------------------------------------------------
with open("chunks_export.json", "r", encoding="utf-8") as f:
    DOCUMENTOS = json.load(f)

TEXTOS = [d["text"] for d in DOCUMENTOS]

print(f"Armando índice TF-IDF con {len(TEXTOS)} fragmentos...")
vectorizer = TfidfVectorizer()
matriz = vectorizer.fit_transform(TEXTOS)
print("Índice TF-IDF listo.")


def buscar_fragmentos(pregunta: str, k: int = 4):
    vector_pregunta = vectorizer.transform([pregunta])
    similitudes = cosine_similarity(vector_pregunta, matriz)[0]
    top_k_idx = similitudes.argsort()[::-1][:k]
    return [DOCUMENTOS[i] for i in top_k_idx]


# ---------------------------------------------------------------------------
# 2. LLM y prompt
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

system_prompt = """Sos el asistente virtual de Mercado Central 24h. Respondé preguntas de colaboradores o clientes
basándote únicamente en el contexto de los documentos internos que se te provee (inventario, política de atención,
FAQ, reglamento interno, manual de proveedores).
Si la respuesta no está en el contexto, decí claramente que no encontraste esa información en los documentos, no inventes datos.
Respondé siempre en español, de forma clara y directa, citando de qué documento sale la información si es relevante.

Contexto:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

chain = prompt | llm


def preguntar(pregunta: str):
    fragmentos = buscar_fragmentos(pregunta)
    contexto = "\n\n".join(f"[{f['source']}]\n{f['text']}" for f in fragmentos)
    respuesta = chain.invoke({"context": contexto, "input": pregunta})

    contenido = respuesta.content
    if isinstance(contenido, list):
        texto = "".join(
            bloque.get("text", "")
            for bloque in contenido
            if isinstance(bloque, dict) and bloque.get("type") == "text"
        )
    else:
        texto = contenido

    fuentes = sorted(set(f["source"] for f in fragmentos))
    return texto, fuentes


print("Agente listo.")

# ---------------------------------------------------------------------------
# Rutas web
# ---------------------------------------------------------------------------
PAGINA_HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Alura Agente — Mercado Central 24h</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
    textarea, input[type=text] { width: 100%; padding: 8px; font-size: 1rem; }
    button { padding: 8px 16px; font-size: 1rem; margin-top: 8px; cursor: pointer; }
    .respuesta { white-space: pre-wrap; background: #f4f4f4; padding: 16px; border-radius: 8px; margin-top: 20px; }
    .fuentes { color: #555; font-size: 0.9rem; margin-top: 10px; }
  </style>
</head>
<body>
  <h1>🛒 Alura Agente — Mercado Central 24h</h1>
  <p>Preguntale al agente sobre el inventario, la política de atención, el reglamento interno o el manual de proveedores.</p>
  <form method="POST" action="/">
    <input type="text" name="pregunta" placeholder="Ej: ¿Cuál es el plazo para devolver carnes y pescados frescos?" value="{{ pregunta or '' }}">
    <button type="submit">Preguntar</button>
  </form>
  {% if respuesta %}
  <div class="respuesta">{{ respuesta }}</div>
  <div class="fuentes">Fuentes usadas: {{ fuentes }}</div>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    respuesta = None
    fuentes = None
    pregunta = None
    if request.method == "POST":
        pregunta = request.form.get("pregunta", "").strip()
        if pregunta:
            respuesta, fuentes_list = preguntar(pregunta)
            fuentes = ", ".join(fuentes_list)
    return render_template_string(PAGINA_HTML, respuesta=respuesta, fuentes=fuentes, pregunta=pregunta)


@app.route("/preguntar", methods=["POST"])
def preguntar_api():
    data = request.get_json(silent=True) or {}
    pregunta = data.get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "Falta el campo 'pregunta' en el body JSON."}), 400

    respuesta, fuentes = preguntar(pregunta)
    return jsonify({"pregunta": pregunta, "respuesta": respuesta, "fuentes": fuentes})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
