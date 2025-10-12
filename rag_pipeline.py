import os
import pandas as pd
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

# --- B. VERİ SETİ OLUŞTURMA ---
def create_and_save_data(filename):
    # Lütfen bu listeyi 100 filme tamamlayınız!
    movie_data = [
        {"title": "The Shawshank Redemption", "genre": "Drama", "year": 1994, "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."},
        {"title": "The Godfather", "genre": "Crime, Drama", "year": 1972, "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
        {"title": "The Dark Knight", "genre": "Action, Crime, Drama", "year": 2008, "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
        {"title": "Pulp Fiction", "genre": "Crime, Drama", "year": 1994, "plot": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
        {"title": "The Lord of the Rings: The Return of the King", "genre": "Adventure, Fantasy", "year": 2003, "plot": "Gandalf and Aragorn lead the World of Men against Sauron's army to give Frodo and Sam a chance to destroy the One Ring. A story of massive battles and final sacrifice."},
        # ... (Lütfen buraya 100 filme tamamlayacak şekilde ekleme yapın)
    ]
    df = pd.DataFrame(movie_data)
    df.to_csv(filename, index=False, encoding='utf-8')

# --- C. RAG PIPELINE KURULUMU ---
def get_qa_chain():
    if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
        raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmalıdır. Streamlit Cloud'da Secrets bölümünden ayarlayın.")

    # 1. Veri Hazırlığı
    create_and_save_data(FILE_NAME) # Uygulama her çalıştığında veriyi oluşturur.
    loader = CSVLoader(file_path=FILE_NAME, encoding="utf-8", csv_args={'delimiter': ','}, source_column="title")
    documents = loader.load()

    # 2. Embedding ve Vektör Veritabanı
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # ChromaDB (Oluştur ve kaydet)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name=COLLECTION_NAME)
    # Streamlit Cloud'da diske yazmak yerine sadece bellekte tutuyoruz, çünkü disk kalıcı değildir.

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
    # Konsol testi
    qa_chain = get_qa_chain()
    test_query = "Karmaşık kurgusu olan, bilim kurgu türünde bir film önerisi yap."
    print(f"TEST EDİLİYOR: {test_query}")
    result = qa_chain.invoke({"query": test_query})
    print("
MODEL CEVABI:")
    print(result["result"])