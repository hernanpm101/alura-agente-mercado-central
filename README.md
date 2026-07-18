# Alura Agente — Mercado Central 24h 🛒🤖

Agente de IA que responde preguntas en lenguaje natural sobre los documentos internos de **Mercado Central 24h**, un supermercado de operación continua (24/7): inventario, política de atención al cliente, FAQ, reglamento interno y manual de proveedores. Cualquier colaborador puede preguntarle directamente en vez de tener que abrir y buscar dentro de cada documento.

Proyecto final del challenge de Alura: lectura de documentos + agente conversacional + deploy en la nube (Render).

## 🏗️ Arquitectura

```
Usuario (pregunta en lenguaje natural)
        │
        ▼
  Notebook (Google Colab)
        │
        ▼
  LangChain — RAG (Retrieval-Augmented Generation)
        │
        ├── 1. Descarga de los documentos (requests)
        │      → 4 PDFs (política de atención, FAQ, reglamento interno, manual de proveedores)
        │      → 1 Excel (inventario de productos)
        │
        ├── 2. Carga y troceo
        │      → PDFs: PyPDFLoader + RecursiveCharacterTextSplitter (chunk_size=2000)
        │      → Excel: agrupado en bloques de 25 filas por documento (Pandas)
        │
        ├── 3. Embeddings LOCALES (sin API, sin cuota, sin costo)
        │      → HuggingFaceEmbeddings
        │        (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
        │
        ├── 4. Indexación en FAISS (búsqueda por similitud semántica)
        │
        ├── 5. Recuperación de los fragmentos más relevantes (top-k=4)
        │
        └── 6. Generación de la respuesta con Gemini 3 Flash (Google)
        │      → create_stuff_documents_chain + create_retrieval_chain (LangChain)
        │
        ▼
  Respuesta en español + fuentes usadas (qué documento la sustenta)
```

### Por qué este stack

- **LangChain**: orquesta la carga, indexación y consulta de los documentos.
- **Embeddings locales (Hugging Face)**: en vez de depender de una API externa de embeddings —cuya cuota gratuita resultó ser el cuello de botella real durante el desarrollo—, el vector de cada fragmento se calcula en la propia máquina, sin llamadas de red ni límites de uso.
- **Gemini 3 Flash**: se usa únicamente para la etapa de generación de la respuesta final. Su cuota gratuita (1500 solicitudes/día) es mucho más generosa que la de embeddings y nunca representó un problema.
- **FAISS**: búsqueda vectorial local y rápida, sin infraestructura adicional.
- **RAG**: el modelo solo responde en base a lo que efectivamente está en los documentos —si la respuesta no está ahí, lo dice explícitamente en vez de inventar datos.

## 📂 Estructura del repositorio

```
alura-agente-mercado-central/
├── Alura_Agente_Mercado_Central.ipynb   # Notebook principal (Google Colab)
├── requirements.txt                      # Dependencias del proyecto
├── .gitignore
└── README.md
```

> Los documentos (PDFs, Excel) y el índice FAISS **no** se versionan en el repo — el propio notebook los descarga y regenera automáticamente en cada ejecución (ver `.gitignore`).

## ▶️ Cómo ejecutar el proyecto

### Google Colab (recomendado)
1. Subí `Alura_Agente_Mercado_Central.ipynb` a [Google Colab](https://colab.research.google.com/).
2. Generá una API Key gratuita de Gemini en [Google AI Studio](https://aistudio.google.com/apikey) (no requiere tarjeta).
3. Corré las celdas **en orden, de la 1 a la 9**, sin saltear ninguna. La celda 5 (creación del índice) tarda ~1 minuto la primera vez porque descarga el modelo de embeddings.
4. Preguntale lo que quieras al agente en las celdas 8 (ejemplos) o 9 (modo interactivo).

### Local
```bash
git clone <URL-de-este-repo>
cd alura-agente-mercado-central
pip install -r requirements.txt
jupyter notebook Alura_Agente_Mercado_Central.ipynb
```

## 💬 Ejemplos de preguntas y respuestas

Estos son ejemplos reales generados por el agente, tal como los devolvió en la ejecución del notebook.

---

> **Pregunta:** ¿Cuál es el plazo para devolver carnes y pescados frescos?
>
> **Respuesta:** De acuerdo con la política de devoluciones de Mercado Central 24h, el plazo para realizar el cambio de carnes y pescados frescos es el mismo día de la compra, antes de las 22:00 horas. Esta información se encuentra detallada en la sección de políticas de productos perecederos del manual de atención al cliente.
>
> *Fuentes usadas: faq.pdf, reglamento_interno.pdf, politica_atencion.pdf*

---

> **Pregunta:** ¿Cuántos días de vacaciones tengo con 5 años de antigüedad?
>
> **Respuesta:** No encontré información específica sobre la cantidad de días de vacaciones que corresponden según los años de antigüedad en los documentos proporcionados. Los documentos del Reglamento Interno (Sección B4. Nómina e Incidencias) solo detallan el procedimiento para solicitarlas, la política de acumulación y que la antigüedad otorga prelación (prioridad) al elegir las fechas, pero no incluyen la tabla de días por año trabajado. Te sugiero consultar directamente con el área de Capital Humano para obtener el dato exacto según tu antigüedad.
>
> *Fuentes usadas: faq.pdf, politica_atencion.pdf*
>
> *(Este ejemplo demuestra que el agente reconoce los límites de su propio conocimiento en vez de inventar una respuesta cuando la información no está en los documentos.)*

---

> **Pregunta:** ¿Cómo funciona el programa Cliente VIP Central y cuáles son sus niveles?
>
> **Respuesta:** El programa "Cliente VIP Central" es un sistema de fidelidad gratuito diseñado para que los clientes acumulen "Pesos Central" por sus compras y accedan a beneficios exclusivos. Cuenta con tres niveles según el monto de compra acumulado:
> - **Nivel Plata (Entrada):** automático al registrarse, sin compra mínima. 2% en "Pesos Central", precios exclusivos, notificaciones anticipadas de ofertas.
> - **Nivel Oro:** requiere acumular $5,000 MXN o más en 3 meses consecutivos. 3.5% en "Pesos Central", fila preferencial en cajas, estacionamiento gratuito, envío a domicilio gratis en compras mayores a $800 MXN.
> - **Nivel Diamante:** requiere acumular $15,000 MXN o más en 3 meses consecutivos. Incluye todos los beneficios anteriores más privilegios adicionales.
>
> El registro puede hacerse en el módulo de Atención a Clientes, la App móvil o el sitio web. El sistema evalúa el nivel de cada cliente mensualmente, y los niveles Oro y Diamante pueden activar una "Cuenta Familiar" para compartir el saldo con hasta 3 tarjetas adicionales.
>
> *Fuentes usadas: faq.pdf, politica_atencion.pdf*

---

> **Pregunta:** hola
>
> **Respuesta:** ¡Hola! Soy el asistente virtual de Mercado Central 24h. ¿En qué puedo ayudarte hoy? Puedo brindarte información basada en nuestros documentos oficiales sobre: Programa VIP (acumulación y canje de puntos, beneficios y vigencia), atención a clientes con necesidades especiales, información corporativa y de contacto, y procedimientos internos.
>
> *Fuentes usadas: politica_atencion.pdf, reglamento_interno.pdf*
>
> *(Ejemplo del modo interactivo del agente — sección 9 del notebook.)*

## ☁️ Deploy en la nube

La consigna del challenge sugiere **OCI Compute** para el deploy, aclarando explícitamente que es una sugerencia y no una obligación ("si contamos con una herramienta que conocemos mejor y que tenga más sentido para nuestro proyecto, podemos usarla"). Se intentó primero con OCI, pero **el proceso de registro de la cuenta gratuita de Oracle Cloud falló repetidamente** (error genérico de creación de cuenta, un problema reportado por numerosos usuarios en foros de Oracle y no atribuible a un error de configuración propio). Tras varios intentos con distintas regiones sin éxito, se optó por desplegar en **Render**, una plataforma cloud con un tier gratuito de registro simple y confiable.

**🔗 Aplicación desplegada:** *(completar con la URL pública, ej. `https://alura-agente-mercado-central.onrender.com`)*

### Cómo se hizo el deploy

1. Se envolvió la lógica del notebook en una aplicación **Flask** (`app.py`), que expone:
   - Una página web simple (`/`) con un formulario para preguntarle al agente.
   - Un endpoint JSON (`/preguntar`) para integraciones programáticas.
2. Se subió el código a este mismo repositorio de GitHub.
3. Se creó un **Web Service** en Render, conectado directamente al repositorio (branch `main`).
4. Configuración usada:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
   - **Variable de entorno:** `GOOGLE_API_KEY` (la API key de Gemini, configurada como secreto en Render, nunca expuesta en el código ni en el repositorio)
5. Al arrancar, la app descarga los documentos, genera los embeddings localmente (sin depender de ninguna cuota externa) y arma el índice — igual que en el notebook.

> ⚠️ **Nota:** al estar en el tier gratuito, la instancia "se duerme" tras un período de inactividad. La primera consulta después de un tiempo sin uso puede tardar 30-90 segundos en responder mientras el servidor arranca y reconstruye el índice — es un comportamiento esperado del plan gratuito, no un error de la aplicación.

*(Acá va la captura de pantalla de la aplicación funcionando en Render.)*

## 🔧 Personalización

Para usar otros documentos, alcanza con cambiar el diccionario `archivos` en la celda 3 (o subirlos manualmente) — el resto del pipeline funciona igual sin cambios de código, siempre que sean PDF o Excel/CSV.

## 🙌 Créditos

Proyecto desarrollado como challenge final del curso de Alura sobre agentes de IA.

