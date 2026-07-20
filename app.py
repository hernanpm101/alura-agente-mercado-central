"""
Alura Agente — Mercado Central 24h
API/Web app que envuelve el agente RAG para poder desplegarlo en la nube.

Al arrancar (una sola vez por proceso):
  1. Descarga los documentos de Mercado Central 24h (PDFs + Excel).
  2. Los carga y trocea.
  3. Genera embeddings locales (Hugging Face, sin cuota) y arma el índice FAISS.
  4. Crea la cadena de preguntas y respuestas con Gemini.

Expone:
  GET  /             -> formulario simple para preguntarle al agente
  POST /preguntar     -> endpoint JSON: {"pregunta": "..."} -> {"respuesta": "...", "fuentes": [...]}
"""

import os
import requests
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. Descargar los documentos (si no están ya en disco)
# ---------------------------------------------------------------------------
ARCHIVOS = {
    "faq.pdf": "https://cdn1.gnarususercontent.com.br/documents/6/internal/c0a37625-c8ee-44f2-885f-7b30480d3017.pdf",
    "politica_atencion.pdf": "https://cdn1.gnarususercontent.com.br/documents/6/internal/b9abdeaf-ffcb-46c4-8e1b-16935a594875.pdf",
    "reglamento_interno.pdf": "https://cdn1.gnarususercontent.com.br/documents/6/internal/ed99ddc6-bead-47fd-886b-c007c7e36885.pdf",
    "manual_proveedores.pdf": "https://cdn1.gnarususercontent.com.br/documents/6/internal/6ce88d4f-3c95-42c3-aef3-c156e859f000.pdf",
    "inventario.xlsx": "https://cdn1.gnarususercontent.com.br/documents/6/internal/e4dc3cc4-55d5-453b-bc68-3f5ef090fc7f.xlsx",
}


def descargar_documentos():
    for nombre, url in ARCHIVOS.items():
        if not os.path.exists(nombre):
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(nombre, "wb") as f:
                f.write(r.content)
            print(f"Descargado: {nombre}")


# ---------------------------------------------------------------------------
# 2. Cargar y trocear los documentos
# ---------------------------------------------------------------------------
def cargar_y_trocear():
    todos_documentos = []

    pdfs = ["faq.pdf", "politica_atencion.pdf", "reglamento_interno.pdf", "manual_proveedores.pdf"]
    for pdf in pdfs:
        todos_documentos.extend(PyPDFLoader(pdf).load())

    df = pd.read_excel("inventario.xlsx")
    GRUPO_FILAS = 25
    for i in range(0, len(df), GRUPO_FILAS):
        bloque = df.iloc[i:i + GRUPO_FILAS]
        todos_documentos.append(
            Document(page_content=bloque.to_string(index=False), metadata={"source": "inventario.xlsx"})
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(todos_documentos)


# ---------------------------------------------------------------------------
# 3 y 4. Embeddings locales + índice FAISS + cadena del agente
# ---------------------------------------------------------------------------
def construir_agente():
    descargar_documentos()
    chunks = cargar_y_trocear()

    embeddings = FastEmbedEmbeddings(model_name="intfloat/multilingual-e5-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
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


print("Inicializando agente (esto puede tardar 1-2 minutos la primera vez)...")
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
