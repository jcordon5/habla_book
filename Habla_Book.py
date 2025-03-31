import streamlit as st

from book_processor import load_book
from vector_store import create_vector_store
from qa_engine import create_qa_chain

st.set_page_config(page_title="Habla Book 📘")

user_api_key = st.sidebar.text_input("🔑 Introduce tu clave de OpenAI (obligatorio)", type="password")

if not user_api_key:
    st.warning("Necesitas introducir tu API Key de OpenAI para usar la app.")
    st.stop()

st.session_state.api_key = user_api_key

if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pages" not in st.session_state:
    st.session_state.pages = None

if "qa" not in st.session_state:
    st.session_state.qa = None

st.title("📘 Habla Book – Pregunta a tus libros")
st.markdown("Sube un libro y haz preguntas contextualizadas sin spoilers.")

uploaded_file = st.file_uploader("Sube tu libro (PDF)", type="pdf")
max_page = st.number_input("¿Hasta qué página has llegado?", min_value=1, value=5)

if st.session_state.chat_history:
    st.markdown("### 🗣️ Conversación")
    for speaker, msg in st.session_state.chat_history:
        avatar = "👤" if speaker == "user" else "📘"
        st.markdown(f"{avatar} **:** {msg}")
    st.markdown("---")


question = st.text_input("Tu pregunta:")
ask_btn = st.button("Preguntar")

system_prompt = (
    "Eres un asistente de lectura que ayuda a responder preguntas sobre un libro. "
    "No harás spoilers, por lo que si no encuentras información sobre lo que se te pregunta, "
    "simplemente di que no puedes responder para no hacer spoilers, que el usuario siga leyendo "
    "y que vuelva a preguntar más tarde cuando haya más información disponible.\n"
)

if uploaded_file and st.session_state.vectorstore is None:
        with st.spinner("Procesando libro..."):
            pages = load_book(uploaded_file)
            st.session_state.pages = pages
            vectorstore = create_vector_store(pages, api_key=st.session_state.api_key)
            st.session_state.vectorstore = vectorstore

if st.session_state.vectorstore is None:
    st.warning("Debes subir un libro antes de poder hacer preguntas.")
    st.stop()

if st.session_state.vectorstore and ask_btn and question:
    with st.spinner("Buscando respuesta..."):
        qa = create_qa_chain(st.session_state.vectorstore, max_page=int(max_page), api_key=st.session_state.api_key)
        full_question = system_prompt + question
        result = qa.invoke({"query": full_question})

    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("ai", result["result"]))

    st.session_state.last_sources = result["source_documents"]

    st.rerun()

if "last_sources" in st.session_state:
    with st.expander("🔎 Fuentes utilizadas"):
        for doc in st.session_state.last_sources:
            st.markdown(f"📄 Página: {doc.metadata.get('page_number')}")
            st.write(doc.page_content)
