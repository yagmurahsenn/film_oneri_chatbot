# 🎬 RAG Tabanlı Film Öneri Asistanı (Retrieval Augmented Generation)

## Projenin Amacı (Kriter 1)

Bu projenin temel amacı, **RAG (Retrieval Augmented Generation)** mimarisini kullanarak geleneksel Büyük Dil Modeli (LLM) kısıtlamalarını aşan, **doğrulanabilir** ve **bağlama dayalı** film önerileri sunan bir chatbot geliştirmektir. Chatbot, kullanıcının sorgularına sadece bizim oluşturduğumuz **özel veri tabanında (100 film)** bulunan bilgilere dayanarak cevap verir. Bu yaklaşım, modelin "uydurma" (halüsinasyon) yapma riskini ortadan kaldırır.

---

## 2 - Veri Seti Hakkında Bilgi (Kriter 2)

Bu RAG sistemi, **100 popüler filmden** oluşan, yapılandırılmış bir CSV dosyası kullanır.

| Alan (Sütun) | Açıklama | RAG Mimarisi İçindeki Rolü |
| :--- | :--- | :--- |
| **title** | Filmin Adı | Retriever sonuçlarında gösterilen ana bilgi. |
| **genre** | Filmin Türü | Tür bazlı (Aksiyon, Dram vb.) sorgular için anahtar. |
| **year** | Yapım Yılı | Ek filtreleme ve bağlam için kullanılır. |
| **plot** | Filmin Kısa Özeti/Konusu | **En Kritik Alan.** Gömme (Embedding) işlemi için kullanılır. Kullanıcı sorusuyla anlamsal eşleştirme yapar. |

**Toplanış/Hazırlanış Metodolojisi:**
Veri seti, projenin kapsamına uygun olarak 100 filmle sınırlandırılmış ve CSV formatına dönüştürülmüştür. Film özetleri (`plot`), LLM'e sunulacak olan **bağlamı (context)** oluşturmak ve ChromaDB'de vektörlenmek üzere hazırlanmıştır.

---

## 4 - Çözüm Mimariniz ve Kullanılan Yöntemler (Kriter 4) 
**(Bu etiketi kendim ekledim, orijinal metinde eksikti)**

Proje, **LangChain** çatısı altında RAG mimarisi üzerine kurulmuştur.

### Mimarinin Akışı

1.  **Sorgu Vektörleme:** Kullanıcının sorusu (`"Karmaşık kurgulu bir bilim kurgu öner."`), **Google `text-embedding-004`** modeliyle sayısal bir vektöre çevrilir.
2.  **Erişim (Retrieval):** Bu vektör, **ChromaDB**'de saklanan 100 filmin vektörleriyle karşılaştırılır ve anlamsal olarak en yakın **3 film bilgisi** geri çağrılır.
3.  **Üretim (Generation):** Geri çağrılan 3 filmin bilgisi (**Context**) ve kullanıcının orijinal sorusu, **Gemini 2.5 Flash** modeline özel bir prompt ile gönderilir.
4.  **Cevap:** Gemini, **yalnızca sağlanan bu Context'i kullanarak** filmi gerekçeli bir şekilde önerir.

### Kullanılan Teknolojiler

| Bileşen | Teknoloji | Versiyon | Amaç |
| :--- | :--- | :--- | :--- |
| **LLM (Generation Model)** | `Gemini 2.5 Flash` | - | Hızlı ve bağlama dayalı cevap üretimi. |
| **Embedding Model** | `Google text-embedding-004` | - | Metinleri anlamsal vektörlere çevirme. |
| **Vektör Veritabanı** | `ChromaDB` | - | Vektörleri saklama ve hızlı arama yapma. |
| **RAG Framework** | `LangChain` | - | Tüm RAG akışını (zincirini) yönetme. |
| **Web Arayüzü** | `Streamlit` | - | Chatbot'u web üzerinden sunma. |

---

## 3 - Kodunuzun Çalışma Kılavuzu (Kriter 3)
**(Bu etiketi kendim ekledim, orijinal metinde eksikti)**

### Dosya Yapısı

* `app.py`: Streamlit web arayüzü ve RAG zinciri çağrısı.
* `rag_pipeline.py`: Veri seti hazırlığı, Embedding, ChromaDB kurulumu ve RAG zincirinin ana mantığı.
* `requirements.txt`: Gerekli kütüphane bağımlılıkları.

### Bağımlılıklar (requirements.txt)

```bash
pydantic==2.5.3
langchain-google-genai
langchain-community
chromadb
pandas
streamlit
