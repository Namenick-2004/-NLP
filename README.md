# 🤖 NLP Buddy – แชตบอต RAG ตอบคำถามเรื่อง NLP

Web Application แชตบอตที่ตอบคำถามเกี่ยวกับ **การประมวลผลภาษาธรรมชาติ (NLP)** จากคลังเอกสารความรู้ภาษาไทย ด้วยเทคนิค **RAG (Retrieval-Augmented Generation)** พัฒนาด้วย Streamlit และ Gemini API

> แบบทดสอบเก็บคะแนนครั้งที่ 2 วิชา NLP

## แนวคิดของ Domain

นักศึกษาวิชา NLP มักต้องย้อนอ่านเอกสารหลายแหล่งเพื่อทบทวนเนื้อหา เช่น การตัดคำไทย Embedding Transformer หรือตัววัดผล NLP Buddy รวบรวมเนื้อหาเหล่านี้ไว้ในคลังความรู้เดียว ให้ถามเป็นภาษาธรรมชาติแล้วได้คำตอบสั้น กระชับ พร้อมระบุว่ามาจากเอกสารใด และตอบว่า "ไม่พบข้อมูล" เมื่อเอกสารไม่มีคำตอบ แทนการเดา

## วิธีใช้งาน

1. เปิดหน้าเว็บแล้วพิมพ์คำถามในช่องแชต หรือกดตัวอย่างคำถามที่แถบด้านข้าง
2. ถามต่อเนื่องได้ ระบบจำประวัติการสนทนาล่าสุด 3 รอบ
3. กดเปิด "📚 เอกสารอ้างอิง" ใต้คำตอบเพื่อดู Chunk ที่ใช้ตอบพร้อมคะแนนความคล้าย
4. กด "ล้างการสนทนา" เพื่อเริ่มใหม่

## โครงสร้างระบบ

```
เอกสาร (data/) → ทำความสะอาด → Chunking → Embedding → FAISS
                                                          │
คำถาม → Embedding → ค้น Top-k ──────────────────────────┘
          │              │ คะแนนต่ำกว่าเกณฑ์ทั้งหมด → "ไม่พบข้อมูล"
          └→ Prompt (บริบท + ประวัติ + คำถาม) → Gemini LLM → คำตอบ + อ้างอิง
```

| ขั้นตอน | รายละเอียด |
|---|---|
| Document Loading | อ่านไฟล์ `.md/.txt` ใน `data/` ลบอักขระควบคุม จัดรูปอักขระไทยด้วย `pythainlp.util.normalize` ลบช่องว่างซ้ำ |
| Chunking | แบ่งตามย่อหน้า/บรรทัด แล้วตัดประโยคด้วย `pythainlp.sent_tokenize` รวมเป็น Chunk ไม่เกิน 500 ตัวอักษร ซ้อนทับ 1 ประโยค และใส่หัวข้อเอกสารนำหน้าทุก Chunk |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (เล็ก รองรับไทย/อังกฤษ 384 มิติ) |
| Vector Search | FAISS `IndexFlatIP` กับเวกเตอร์ที่ normalize แล้ว (เท่ากับ cosine similarity) ดึง Top-4 |
| Prompt | สั่งให้ตอบจากบริบทเท่านั้น อ้างอิง `[1] [2]` และตอบ "ไม่พบข้อมูลในเอกสารที่มี" เมื่อไม่มีคำตอบ |
| Guardrail | ถ้าทุก Chunk มีคะแนนต่ำกว่า 0.30 ระบบปฏิเสธทันทีโดยไม่เรียก LLM |
| LLM | Gemini API โมเดล `gemini-3.5-flash-lite` temperature 0.1 |
| Chat UI | `st.chat_message` + `session_state` คุยต่อเนื่องได้ และแสดงเอกสารอ้างอิงทุกคำตอบ |

โมเดล Embedding และ Index ถูกโหลดเพียงครั้งเดียวด้วย `st.cache_resource` เพื่อประหยัดหน่วยความจำบน Streamlit Community Cloud

## ติดตั้งและรันในเครื่อง

```bash
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # แล้วแก้ใส่ GEMINI_API_KEY ของคุณ
streamlit run app.py
```

สร้าง API Key ได้ที่ https://aistudio.google.com/apikey

## Deploy บน Streamlit Community Cloud

1. สร้าง GitHub Repository (Private) แล้วอัปโหลดโค้ดทั้งหมด **ห้ามอัปโหลด `.streamlit/secrets.toml`** (อยู่ใน `.gitignore` แล้ว)
2. ไปที่ https://share.streamlit.io เลือก New app ชี้ไปที่ repo, branch `main`, ไฟล์ `app.py`
3. ที่ Advanced settings → Secrets ใส่
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
4. กด Deploy แล้วทดสอบเปิด URL ในโหมด Incognito

## โครงสร้างไฟล์

```
.
├── app.py                 # แอปหลัก
├── requirements.txt
├── README.md
├── test_questions.csv     # คำถามทดสอบ 14 ข้อ (ตอบได้ 11 / ไม่มีในเอกสาร 3)
├── data/                  # คลังความรู้ 12 ไฟล์
└── .streamlit/secrets.toml.example
```

## แหล่งที่มาของเอกสาร

เอกสารใน `data/` ทั้ง 12 ไฟล์ เรียบเรียงเป็นภาษาไทยโดยใช้ AI (Claude) ช่วยสร้างเนื้อหา จากความรู้พื้นฐานด้าน NLP ที่เป็นที่รู้จักทั่วไป อ้างอิงแนวคิดจากงานต้นฉบับ เช่น Attention Is All You Need (Vaswani et al., 2017), BERT (Devlin et al., 2018), Word2Vec (Mikolov et al., 2013), RAG (Lewis et al., 2020), WangchanBERTa (Lowphansirikul et al., 2021) และเอกสารของ PyThaiNLP, FAISS, Gemini ควรตรวจสอบข้อเท็จจริงกับแหล่งต้นฉบับก่อนนำไปอ้างอิงจริง

| ไฟล์ | หัวข้อ |
|---|---|
| 01 | NLP คืออะไร |
| 02 | การตัดคำภาษาไทย |
| 03 | การเตรียมข้อความ |
| 04 | Word / Sentence Embedding |
| 05 | Transformer และ Attention |
| 06 | BERT และ WangchanBERTa |
| 07 | การจำแนกข้อความและวิเคราะห์ความรู้สึก |
| 08 | NER และ POS Tagging |
| 09 | การแปลภาษาและสรุปความ |
| 10 | RAG |
| 11 | ตัววัดประสิทธิภาพ |
| 12 | LLM และ Prompt Engineering |

## ตัวอย่าง Prompt ที่ใช้สั่ง AI สร้างโปรเจกต์

1. "สร้าง Web Application แชตบอต RAG ด้วย Streamlit ตอบคำถามจากเอกสารภาษาไทย ใช้ Sentence Embedding + FAISS + Gemini API ต้องมี Chunking, แสดงเอกสารอ้างอิงทุกคำตอบ, ตอบว่าไม่พบข้อมูลเมื่อเอกสารไม่มีคำตอบ และเก็บ API Key ใน st.secrets"
2. "เขียนเอกสารความรู้ภาษาไทยเรื่องการประมวลผลภาษาธรรมชาติ 12 หัวข้อ แต่ละไฟล์ยาว 1,500 ตัวอักษรขึ้นไป เนื้อหาถูกต้อง เหมาะสำหรับนักศึกษา"
3. "สร้าง test_questions.csv อย่างน้อย 10 ข้อ พร้อมคำตอบที่ถูกต้อง และมีคำถามที่ไม่มีคำตอบในเอกสารอย่างน้อย 2 ข้อ"

### Prompt ที่ใช้ภายในระบบ (System Prompt)

ดูตัวแปร `SYSTEM_PROMPT` ใน `app.py` กฎสำคัญคือ ใช้บริบทเท่านั้น, อ้างอิง `[n]`, และตอบ "ไม่พบข้อมูลในเอกสารที่มี" เมื่อไม่มีข้อมูล
