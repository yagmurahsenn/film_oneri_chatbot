import os
import pandas as pd
import json
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- A. GENEL AYARLAR ---
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "text-embedding-004" 
FILE_NAME = "movies_100.csv"
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "movie_recommendations"


# --- B. VERİ SETİ OLUŞTURMA (JSON'dan okuma) ---
# Bu fonksiyon, Streamlit Cloud'da JSON dosyanızdaki 100 filmi okuyup CSV'ye çevirir.
def create_and_save_data(filename):
    # movie_data.json dosyasını okur
    try:
        with open("movie_data.json", "r", encoding="utf-8") as f:
            movie_data_list = json.load(f)
    except FileNotFoundError:
        # Eğer JSON dosyası bulunamazsa, uygulama burada durur.
        raise FileNotFoundError("movie_data.json dosyası bulunamadı. Lütfen GitHub'a yüklediğinizden emin olun.")
        
    # CSV'ye kaydeder
    df_movie = pd.DataFrame(movie_data_list)
    df_movie.to_csv(filename, index=False, encoding='utf-8')

    # Veri setini kontrol etme
    if 'title' not in df_movie.columns:
        raise ValueError("Veri setinde 'title' sütunu bulunamadı. JSON dosyanızın formatı hatalı.")


# --- C. RAG PIPELINE KURULUMU ---
# Bu fonksiyon, Streamlit'in @st.cache_resource ile bir kez çağıracağı fonksiyondur.
def get_qa_chain():
    # API Anahtarı kontrolü
    if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
        raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmalıdır. Streamlit Cloud'da Secrets bölümünden ayarlayın.")

    # 1. Veri Hazırlığı
    create_and_save_data(FILE_NAME) 
    loader = CSVLoader(file_path=FILE_NAME, encoding="utf-8", csv_args={'delimiter': ','}, source_column="title")
    documents = loader.load()

    # 2. Embedding ve Vektör Veritabanı
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name=COLLECTION_NAME)

    # 3. RAG Zincirini Kurma
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2) 
    
    prompt_template = """Sen, yalnızca sana sağlanan film veritabanından öneri yapabilen bir film öneri asistanısın.
Kullanıcıya nazikçe, akıcı bir dille cevap ver. Önerdiğin filmlerin özetini ve türünü açıklayarak önerini gerekçelendir.
Eğer verilen filmler (CONTEXT) kullanıcının sorusuna uygun değilse, kibarca "Üzgünüm, veri tabanımda bu kritere uygun bir film bulamadım." diye cevap ver.
Cevaplarında önerdiğin film adı, türü ve yılı mutlaka yer almalıdır.

CONTEXT (ChromaDB'den gelen en ilgili filmler):
{context}

Kullanıcının Sorusu: {question}

Cevabın:"""

    RAG_PROMPT_GEMINI = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": RAG_PROMPT_GEMINI}
    )
    return qa_chain

if __name__ == "__main__":
    # Konsolda RAG zincirini hızlıca test etmek için
    qa_chain = get_qa_chain()
    test_query = "İnsan psikolojisi temalı, dramatik ve derin bir film önerisi yap."
    print(f"TEST EDİLİYOR: {test_query}")
    result = qa_chain.invoke({"query": test_query})
    print("\nMODEL CEVABI:")
    print(result["result"])
