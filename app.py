import streamlit as st
import os
from rag_pipeline import get_qa_chain

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    
    os.environ['GEMINI_API_KEY'] = api_key
    os.environ['GOOGLE_API_KEY'] = api_key 
    
except KeyError:
    st.error("Kurulum hatası: Lütfen Streamlit Secrets bölümünde GEMINI_API_KEY'i ayarlayın.")
    st.stop()
except Exception as e:
    st.error(f"Beklenmedik bir hata oluştu: {e}")
    st.stop()


# --- RAG ZİNCİRİNİ YÜKLEME ---
@st.cache_resource
def load_rag_chain():
    return get_qa_chain()

try:
    qa_chain = load_rag_chain()
    st.success("✅ Film Veri Tabanı Yüklendi! Chatbot hazır.") 
except Exception as e:
    st.error(f"RAG zinciri kurulumunda beklenmeyen bir hata oluştu: {e}")
    st.stop()


# --- STREAMLIT ARAYÜZÜ ---
st.title("🎬 RAG Tabanlı Film Öneri Asistanı")
st.caption("Gemini, LangChain ve ChromaDB kullanılarak geliştirilmiştir.")

# Sohbet geçmişini başlatma
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Merhaba! Ben film öneri asistanınızım. Nasıl bir film izlemek istersiniz?"}]

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
