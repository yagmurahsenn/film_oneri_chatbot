# 🎬 RAG Tabanlı Film Öneri Asistanı (Retrieval Augmented Generation)

## 1 - Projenin Amacı 

Bu projenin temel amacı, **RAG ** mimarisini kullanarak geleneksel Büyük Dil Modeli (LLM) kullanan film önerileri sunan bir chatbot geliştirmektir. Chatbot, kullanıcının sorgularına sadece bizim oluşturduğumuz **özel veri tabanında (100 film)** bulunan bilgilere dayanarak cevap verir. Böylece model uydurma cevaplar vermez. 

---

## 2 - Veri Seti Hakkında Bilgi 

Bu RAG sistemi, **100 popüler filmden** oluşan, yapılandırılmış bir CSV dosyası kullanır.

| Alan (Sütun) | Açıklama | RAG Mimarisi İçindeki Rolü |
| :--- | :--- | :--- |
| **title** | Filmin Adı | Retriever sonuçlarında gösterilen ana bilgi. |
| **genre** | Filmin Türü | Tür bazlı (Aksiyon, Dram vb.) sorgular için anahtar. |
| **year** | Yapım Yılı | Ek filtreleme ve bağlam için kullanılır. |
| **plot** | Filmin Kısa Özeti/Konusu | **En Kritik Alan.** Gömme (Embedding) işlemi için kullanılır. Kullanıcı sorusuyla anlamsal eşleştirme yapar. |

**Toplanış/Hazırlanış Metodolojisi:**
Veri seti, projenin kapsamına uygun olarak 100 filmle sınırlandırılmış ve CSV formatına dönüştürülmüştür. Film özetleri (`plot`), LLM'e sunulacak olan **bağlamı (context)** oluşturmak ve ChromaDB'de vektörlenmek için hazırlandı.

---

## 3 - Çözüm Mimarisi ve Kullanılan Yöntemler 

Proje, **LangChain** çatısı altında RAG mimarisi üzerine kurulmuştur.

### Mimarinin Akışı

1.  **Sorgu Vektörleme:** Kullanıcının sorusu (`"Karmaşık kurgulu bir bilim kurgu öner."`), **Google `models/text-embedding-004`** ile sayısal bir vektöre çevrilir.
2.  **Erişim (Retrieval):** Bu vektör, **ChromaDB**'de olan 100 filmin vektörleriyle karşılaştırılır ve en yakın **3 film bilgisi** geri çağrılır.
3.  **Üretim (Generation):** Geri çağrılan 3 filmin bilgisi (**Context**) ve kullanıcının orijinal sorusu, **Gemini 2.5 Flash** modeline bir prompt ile gönderilir.
4.  **Cevap:** Gemini, **yalnızca sağlanan bu Context'i kullanarak** filmi gerekçesiyle önerir.

### Kullanılan Teknolojiler

| Bileşen | Teknoloji | Versiyon | Amaç |
| :--- | :--- | :--- | :--- |
| **LLM (Generation Model)** | `Gemini 2.5 Flash` | - | Hızlı ve bağlama dayalı cevap üretimi. |
| **Embedding Model** | `Google text-embedding-004` | - | Metinleri anlamsal vektörlere çevirme. |
| **Vektör Veritabanı** | `ChromaDB` | - | Vektörleri saklama ve hızlı arama yapma. |
| **RAG Framework** | `LangChain` | - | Tüm RAG akışını (zincirini) yönetme. |
| **Web Arayüzü** | `Streamlit` | - | Chatbot'u web üzerinden sunma. |

---

## 4 - Kodun Çalışma Kılavuzu 

### Dosya Yapısı

* `app.py`: Streamlit web arayüzü ve RAG zinciri çağrısı.
* `rag_pipeline.py`: Veri seti hazırlığı, Embedding, ChromaDB kurulumu ve RAG zincirinin ana mantığı.
* `requirements.txt`: Gerekli kütüphane bağımlılıkları.


Yerel Çalıştırma Adımları
Sanal Ortam Kurulumu:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
API Anahtarının Ayarlanması: (Terminalde, her oturumda yapın)

export GEMINI_API_KEY="[GEMINI API ANAHTARINIZI BURAYA YAZIN]"

Projenin Başlatılması:
streamlit run app.py

### 5 - Requirements.txt

pydantic==2.5.3

langchain-google-genai

langchain-community

chromadb

pandas

streamlit

Elde Edilen Sonuçlar Özeti 
Geliştirilen RAG sistemi, 100 filmlik veri setine bağlı kalarak isabetli ve gerekçeli film önerileri sunabilmektedir. Chatbot, sadece film adına değil, aynı zamanda film özetine(plot) göre de eşleşmeler yapabilmektedir.

Web Linki (Deployment)
Projenin canlı web adresi aşağıdadır. 

Canlı Uygulama: https://filmonerichatbot-mm26glrrxyv4dhre3hvzjb.streamlit.app/

**Projenin Çalışma Demosu:**
Canlı chatbot'un soru-cevap yeteneklerini gösteren kısa video aşağıdadır.

[Proje Demo Videosu](https://youtube.com/shorts/OMypliw_saA?si=Jala-01ncC1pFwlR)


Test Senaryoları 
Canlı arayüze giderek aşağıdaki örnek sorgularla chatbot'un doğru çalışıp çalışmadığını test edebilirsiniz:

Sorgu: "Psikolojik gerilim içeren, kurgusu şaşırtıcı bir film önerisi yapar mısın?"

Sorgu: "Savaşın zorluklarını anlatan, dram türünde dokunaklı bir film istiyorum."

Sorgu: "En son çıkan 2024 yapımı süper kahraman filmi hangisi?" (Bu, veri tabanınızda olmadığı için reddedilmelidir.)

