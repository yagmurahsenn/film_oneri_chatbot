import streamlit as st
import os
from rag_pipeline import get_qa_chain

# --- 🚨 KRİTİK DÜZELTME: API Anahtarını Secrets'ten Ortam Değişkenine Aktarma ---
# Bu, get_qa_chain() fonksiyonunun API anahtarını bulmasını sağlar.
# GOOGLE_API_KEY'i de ekliyoruz çünkü LangChain Embedding modeli bunu arıyor.
if 'GEMINI_API_KEY' in st.secrets:
    os.environ['GEMINI_API_KEY'] = st.secrets['GEMINI_API_KEY']
    os.environ['GOOGLE_API_KEY'] = st.secrets['GEMINI_API_KEY'] 
# ----------------------------------------------------------------------------


# --- RAG ZİNCİRİNİ YÜKLEME ---
@st.cache_resource
def load_rag_chain():
    return get_qa_chain()

try:
    qa_chain = load_rag_chain()
except ValueError as e:
    # Bu hata, rag_pipeline.py dosyasındaki API kontrolünden gelir.
    st.error(f"Kurulum hatası: {e}. Lütfen Streamlit Secrets bölümünde GEMINI_API_KEY'i doğru ayarladığınızdan emin olun.")
    st.stop()
except Exception as e:
    # Diğer beklenmedik hatalar
    st.error(f"Beklenmedik RAG kurulum hatası: {e}")
    st.stop()
try:
    qa_chain = load_rag_chain()
except ValueError as e:
    # ... (Hata kodu aynı kalır)

st.success("✅ Film Veri Tabanı Yüklendi! Chatbot hazır.") 
# --------------------------------

# --- STREAMLIT ARAYÜZÜ ---
st.title("🎬 RAG Tabanlı Film Öneri Asistanı")
# ... (Geri kalan kod aynı)

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
            st.write(response["result"])
