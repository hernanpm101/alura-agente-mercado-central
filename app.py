"""
Alura Agente — Mercado Central 24h
API/Web app que envuelve el agente RAG para poder desplegarlo en la nube.

El índice FAISS ya viene PRECALCULADO (carpeta faiss_index_deploy/, generada en Colab
y subida junto con el código) para evitar picos de memoria al generar cientos de
embeddings en el momento del deploy — el free tier de la nube tiene poca RAM disponible.
En producción, la app solo carga el índice y genera el embedding de cada pregunta
individual, que es mucho más liviano.

Expone:
  GET  /             -> formulario simple para preguntarle al agente
  POST /preguntar     -> endpoint JSON: {"pregunta": "..."} -> {"respuesta": "...", "fuentes": [...]}
"""

import os
from flask import Flask, request, jsonify, render_template_string

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

INDEX_PATH = "faiss_index_deploy"


def construir_agente():
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

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

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


print("Cargando índice precalculado y armando el agente...")
qa_chain = construir_agente()
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
            resultado = qa_chain.invoke({"input": pregunta})
            respuesta = resultado["answer"]
            fuentes = ", ".join(sorted(set(doc.metadata.get("source", "desconocida") for doc in resultado["context"])))
    return render_template_string(PAGINA_HTML, respuesta=respuesta, fuentes=fuentes, pregunta=pregunta)


@app.route("/preguntar", methods=["POST"])
def preguntar_api():
    data = request.get_json(silent=True) or {}
    pregunta = data.get("pregunta", "").strip()
    if not pregunta:
        return jsonify({"error": "Falta el campo 'pregunta' en el body JSON."}), 400

    resultado = qa_chain.invoke({"input": pregunta})
    fuentes = sorted(set(doc.metadata.get("source", "desconocida") for doc in resultado["context"]))
    return jsonify({"pregunta": pregunta, "respuesta": resultado["answer"], "fuentes": fuentes})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
