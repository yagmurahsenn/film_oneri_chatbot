import streamlit as st
import os
from rag_pipeline import get_qa_chain

# --- RAG ZİNCİRİNİ YÜKLEME ---
# @st.cache_resource, RAG kurulumunu (Embedding ve Vektörleme) sadece bir kez yapar.
@st.cache_resource
def load_rag_chain():
    return get_qa_chain()

try:
    qa_chain = load_rag_chain()
except ValueError as e:
    # API Anahtarı eksikse uyarı verir.
    st.error(f"Kurulum hatası: {e}. Lütfen Streamlit Secrets bölümünde GEMINI_API_KEY'i ayarlayın.")
    st.stop()


# --- STREAMLIT ARAYÜZÜ ---
st.title("🎬 RAG Tabanlı Film Öneri Asistanı")
st.caption("Gemini, LangChain ve ChromaDB kullanılarak geliştirilmiştir.")

# Sohbet geçmişini başlatma
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Merhaba! Ben film öneri asistanınızım. Nasıl bir film izlemek istersiniz? (Örn: Dram, bilim kurgu, aksiyon)"}]

# Geçmiş mesajları görüntüleme
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Kullanıcıdan girdi alma
if prompt := st.chat_input("Film önerisi isteyin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Öneri aranıyor..."):
            response = qa_chain.invoke({"query": prompt})
            
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})
            st.write(response["result"])