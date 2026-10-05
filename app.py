"""NLP Buddy - แชตบอต RAG ตอบคำถามเรื่องการประมวลผลภาษาธรรมชาติ (NLP)

Pipeline: โหลดเอกสาร -> ทำความสะอาด -> Chunking -> Embedding -> FAISS
          -> Retrieval -> Prompt -> Gemini LLM -> แสดงคำตอบ + แหล่งอ้างอิง
"""
import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pythainlp.tokenize import sent_tokenize
from pythainlp.util import normalize
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------- ค่าตั้งต้น
DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # เล็ก รองรับภาษาไทย
LLM_MODEL = "gemini-3.5-flash-lite"
CHUNK_SIZE = 500        # จำนวนตัวอักษรสูงสุดต่อ chunk
CHUNK_OVERLAP = 1       # จำนวนประโยคที่ซ้อนทับกับ chunk ก่อนหน้า
TOP_K = 4
MIN_SCORE = 0.30        # ต่ำกว่านี้ถือว่าเอกสารไม่เกี่ยวข้อง -> ไม่ต้องเรียก LLM
HISTORY_TURNS = 3       # จำนวนรอบสนทนาล่าสุดที่ส่งให้ LLM
NOT_FOUND = "ไม่พบข้อมูลในเอกสารที่มี"
ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');

:root {
    --buddy-ink: var(--text-color);
    --buddy-muted: color-mix(in srgb, var(--text-color) 68%, transparent);
    --buddy-primary: #2563eb;
    --buddy-cyan: #0891b2;
    --buddy-line: rgba(37, 99, 235, 0.15);
}

html, body, .stApp, button, input, textarea { font-family: 'Noto Sans Thai', sans-serif; }
/* ห้ามทับฟอนต์ Material Symbols ไม่เช่นนั้นไอคอนจะแสดงเป็นคำว่า face/smart_toy */
.material-symbols-rounded,
.material-symbols-outlined,
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
}
.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(37,99,235,.08), transparent 28rem),
        radial-gradient(circle at 92% 8%, rgba(8,145,178,.07), transparent 24rem),
        var(--background-color);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220 0%, #111c33 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color: #f4f6ff; }
[data-testid="stSidebar"] code {
    color: #d8dcff;
    background: rgba(255,255,255,.09);
}
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 12px;
    text-align: left;
    transition: all .18s ease;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    background: rgba(37,99,235,.24);
    border-color: rgba(96,165,250,.62);
    transform: translateY(-1px);
}
.block-container { max-width: 1100px; padding-top: 1.6rem; padding-bottom: 7rem; }
.buddy-hero {
    padding: 2rem;
    border-radius: 28px;
    background: linear-gradient(130deg, #0f172a 0%, #1d4ed8 56%, #07849b 100%);
    color: white;
    box-shadow: 0 24px 65px rgba(15,48,105,.24);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.buddy-hero:before, .buddy-hero:after {
    content: '';
    position: absolute; border-radius: 50%; pointer-events: none;
}
.buddy-hero:before { width: 260px; height: 260px; right: -85px; top: -120px; background: rgba(255,255,255,.10); }
.buddy-hero:after { width: 160px; height: 160px; left: 43%; bottom: -120px; background: rgba(60,255,222,.15); filter: blur(4px); }
.buddy-hero-inner { position: relative; z-index: 1; display: grid; grid-template-columns: 1.35fr .85fr; gap: 1.8rem; align-items: center; }
.buddy-kicker { font-size: .72rem; font-weight: 700; letter-spacing: .15em; color: #bae6fd; }
.buddy-hero h1 { color: white; font-size: 2.65rem; margin: .35rem 0 .45rem; line-height: 1.15; letter-spacing: -.03em; }
.buddy-hero p { color: rgba(255,255,255,.82); margin: 0; max-width: 610px; line-height: 1.65; }
.buddy-tags { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: 1rem; }
.buddy-tag {
    padding: .34rem .72rem; border-radius: 999px; font-size: .73rem; font-weight: 600;
    background: rgba(255,255,255,.11); border: 1px solid rgba(255,255,255,.18); backdrop-filter: blur(8px);
}
.buddy-live { display: inline-flex; align-items: center; gap: .4rem; padding: .3rem .62rem; border-radius: 999px; background: rgba(9,30,50,.24); border: 1px solid rgba(255,255,255,.14); font-size: .72rem; margin-bottom: .65rem; }
.buddy-live-dot { width: 7px; height: 7px; background: #5df2c4; border-radius: 50%; box-shadow: 0 0 12px #5df2c4; }
.rag-visual { padding: 1rem; border-radius: 20px; background: rgba(12,20,57,.28); border: 1px solid rgba(255,255,255,.14); backdrop-filter: blur(12px); }
.rag-visual-title { font-size: .68rem; font-weight: 700; letter-spacing: .11em; color: rgba(255,255,255,.62); margin-bottom: .65rem; }
.rag-flow { display: grid; gap: .5rem; }
.rag-step { display: flex; align-items: center; gap: .65rem; padding: .55rem .65rem; border-radius: 12px; background: rgba(255,255,255,.08); }
.rag-step-icon { display: grid; place-items: center; width: 30px; height: 30px; border-radius: 9px; background: rgba(255,255,255,.13); font-size: .9rem; }
.rag-step-text b { display: block; color: white; font-size: .78rem; }
.rag-step-text span { display: block; color: rgba(255,255,255,.58); font-size: .66rem; }
.buddy-section-label { color: var(--buddy-primary); font-size: .72rem; font-weight: 700; letter-spacing: .12em; margin-bottom: .2rem; }
.buddy-section-title { color: var(--buddy-ink); font-size: 1.25rem; font-weight: 700; margin-bottom: .15rem; }
.buddy-section-sub { color: var(--buddy-muted); font-size: .86rem; }
.buddy-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
.buddy-card {
    background: color-mix(in srgb, var(--secondary-background-color) 88%, transparent); border: 1px solid var(--buddy-line);
    border-radius: 19px; padding: 1.15rem; box-shadow: 0 10px 30px rgba(46,54,91,.07); position: relative; overflow: hidden;
    transition: transform .2s ease, box-shadow .2s ease;
}
.buddy-card:before { content: ''; position: absolute; inset: 0 0 auto; height: 3px; background: linear-gradient(90deg, #2563eb, #06b6d4); }
.buddy-card:hover { transform: translateY(-3px); box-shadow: 0 16px 38px rgba(15,48,105,.12); }
.buddy-card-icon { display: grid; place-items: center; width: 40px; height: 40px; border-radius: 12px; font-size: 1.2rem; background: rgba(37,99,235,.10); }
.buddy-card strong { color: var(--buddy-ink); display: block; margin: .35rem 0 .15rem; }
.buddy-card span { color: var(--buddy-muted); font-size: .84rem; line-height: 1.55; }
.buddy-note {
    border: 1px solid rgba(8,145,178,.18); background: rgba(8,145,178,.07);
    color: var(--text-color); border-radius: 14px; padding: .8rem 1rem; font-size: .86rem;
}
[data-testid="stChatMessage"] {
    background: color-mix(in srgb, var(--secondary-background-color) 82%, transparent); border: 1px solid rgba(37,99,235,.14);
    border-radius: 20px; padding: .5rem .7rem; box-shadow: 0 8px 26px rgba(15,48,105,.06); margin-bottom: .7rem;
}
[data-testid="stChatMessageAvatarUser"] { background: linear-gradient(135deg, #2563eb, #3b82f6) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: linear-gradient(135deg, #0891b2, #06b6d4) !important; }
[data-testid="stChatInput"] { border: 1px solid rgba(37,99,235,.22); border-radius: 18px; box-shadow: 0 12px 40px rgba(15,48,105,.13); }
[data-testid="stExpander"] {
    background: color-mix(in srgb, var(--secondary-background-color) 74%, transparent); border-color: rgba(37,99,235,.17); border-radius: 14px;
}
.buddy-brand { padding: .35rem 0 1rem; }
.buddy-brand-title { font-size: 1.35rem; font-weight: 700; letter-spacing: -.02em; }
.buddy-brand-sub { font-size: .78rem; opacity: .65; }
.buddy-status {
    display: inline-flex; align-items: center; gap: .4rem; font-size: .78rem;
    padding: .3rem .6rem; border-radius: 999px; background: rgba(33,208,154,.13);
    border: 1px solid rgba(33,208,154,.28); margin-bottom: .8rem;
}
.buddy-dot { width: 7px; height: 7px; border-radius: 50%; background: #20d09a; box-shadow: 0 0 9px #20d09a; }
[data-testid="stSidebar"] [data-testid="stMetric"] { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: .55rem .65rem; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.08); }
@media (max-width: 720px) {
    .block-container { padding-top: 1rem; }
    .buddy-hero { padding: 1.2rem; border-radius: 17px; }
    .buddy-hero h1 { font-size: 1.75rem; }
    .buddy-hero-inner { grid-template-columns: 1fr; }
    .rag-visual { display: none; }
    .buddy-grid { grid-template-columns: 1fr; }
}
</style>
"""

SYSTEM_PROMPT = """คุณคือ "NLP Buddy" ผู้ช่วยตอบคำถามเรื่องการประมวลผลภาษาธรรมชาติ (NLP)
ตอบโดยอิงจาก "บริบท" ที่ให้มาเท่านั้น

กฎ:
1. ใช้ข้อมูลจากบริบทเท่านั้น ห้ามใช้ความรู้ภายนอกหรือเดาเพิ่ม
2. อ้างอิงแหล่งที่มาท้ายประโยคที่ใช้ข้อมูล ในรูปแบบ [1], [2] ตามหมายเลขของบริบท
3. ถ้าบริบทไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบเพียงว่า "ไม่พบข้อมูลในเอกสารที่มี" ห้ามแต่งคำตอบ
4. ตอบเป็นภาษาเดียวกับที่ผู้ใช้ถาม (ไทยหรืออังกฤษ) กระชับ ชัดเจน
5. ใช้ประวัติการสนทนาเพื่อเข้าใจคำถามต่อเนื่อง แต่ข้อเท็จจริงต้องมาจากบริบทเท่านั้น
6. ตรวจชื่อเฉพาะและคำย่อในคำถามให้ตรงกับบริบท ห้ามแก้คำผิดเป็นอีกคำแล้วตอบแทน เช่น NLB ไม่ใช่ NLP
7. หากมีคำเตือนว่าพบคำย่อหรือชื่อเฉพาะที่ไม่มีในคลัง ให้วิเคราะห์ก่อนตอบ และตอบว่าไม่พบข้อมูลถ้าคำนั้นไม่ปรากฏในบริบท"""


# ---------------------------------------------------- 1) Document Loading
def clean_text(text: str) -> str:
    """ทำความสะอาดข้อความ: ลบอักขระควบคุม, จัดรูปอักขระไทย, ลบช่องว่างซ้ำ"""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # แก้สระ/วรรณยุกต์ซ้ำหรือสลับตำแหน่ง (ทำทีละบรรทัด เพราะ normalize จะยุบบรรทัดว่าง)
    text = "\n".join(normalize(line) for line in text.split("\n"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.append({"source": path.name,
                         "text": clean_text(path.read_text(encoding="utf-8"))})
    return docs


# ---------------------------------------------------------- 2) Chunking
def split_sentences(paragraph: str) -> list[str]:
    """ตัดประโยคด้วย PyThaiNLP (ถ้าล้มเหลวให้ใช้ย่อหน้าทั้งก้อน)"""
    try:
        sents = [s.strip() for s in sent_tokenize(paragraph) if s.strip()]
    except Exception:
        sents = []
    return sents or [paragraph]


def chunk_document(doc: dict) -> list[dict]:
    """แบ่งเอกสารตามย่อหน้า -> ประโยค แล้วรวมให้ไม่เกิน CHUNK_SIZE
    พร้อมซ้อนทับ CHUNK_OVERLAP ประโยค และใส่หัวข้อเอกสารนำหน้าทุก chunk"""
    lines = doc["text"].split("\n")
    title = lines[0].lstrip("# ").strip() if lines else doc["source"]

    sentences = []
    for line in lines:  # ย่อหน้า/หัวข้อย่อย/bullet แต่ละบรรทัดถือเป็นหน่วยแยก
        line = line.strip()
        if not line or line.startswith("# "):  # ข้ามหัวข้อหลัก (ใช้เป็น title แล้ว)
            continue
        sentences.extend(split_sentences(line.lstrip("#").strip()))

    chunks, current = [], []
    for sent in sentences:
        if current and sum(len(s) for s in current) + len(sent) > CHUNK_SIZE:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else []
        current.append(sent)
    if current:
        chunks.append(current)

    return [
        {"source": doc["source"], "title": title, "chunk_id": i,
         "text": f"{title}: " + " ".join(c)}
        for i, c in enumerate(chunks)
    ]


# ------------------------------------------- 3) Embedding & Vector Search
@st.cache_resource(show_spinner="กำลังโหลดโมเดล Embedding...")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="กำลังสร้างดัชนีเอกสาร...")
def build_index():
    embedder = load_embedder()
    chunks = [c for d in load_documents(DATA_DIR) for c in chunk_document(d)]
    vectors = embedder.encode([c["text"] for c in chunks],
                              normalize_embeddings=True,
                              show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # normalize แล้ว IP = cosine
    index.add(vectors)
    return index, chunks


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    index, chunks = build_index()
    q = load_embedder().encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [{**chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0]) if i != -1]


def unknown_acronyms(query: str) -> list[str]:
    """ป้องกันคำย่อที่พิมพ์ผิด เช่น NLB ถูกเข้าใจผิดเป็น NLP โดย Embedding"""
    terms = set(ACRONYM_PATTERN.findall(query))
    if not terms:
        return []
    corpus = "\n".join(
        document["text"] for document in load_documents(DATA_DIR)
    )
    return sorted(
        term for term in terms
        if not re.search(rf"(?<![A-Za-z0-9-]){re.escape(term)}(?![A-Za-z0-9-])",
                         corpus, flags=re.IGNORECASE)
    )


# ------------------------------------------------ 4) Prompt + 5) LLM
def build_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (แหล่งที่มา: {h['source']})\n{h['text']}"
                       for i, h in enumerate(hits, 1))


def retrieval_query(question: str, history: list[dict]) -> str:
    """คำถามต่อเนื่องที่สั้นมาก (เช่น 'แล้วอันนั้นล่ะ') ให้ต่อด้วยคำถามก่อนหน้า"""
    prev = [m["content"] for m in history if m["role"] == "user"]
    if prev and len(question) < 25:
        return f"{prev[-1]} {question}"
    return question


def ask_llm(question: str, hits: list[dict], history: list[dict],
            query_warnings: list[str] | None = None) -> str:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    recent_history = history[-HISTORY_TURNS * 2:]
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in recent_history
    ) or "(ไม่มีประวัติการสนทนา)"
    warning_text = (
        "\n\nคำเตือนจากระบบ: พบคำย่อ/ชื่อเฉพาะที่ไม่ปรากฏแบบตรงตัวในคลัง: "
        + ", ".join(query_warnings)
        + " ห้ามเดาหรือแก้เป็นคำอื่นเอง"
    ) if query_warnings else ""
    prompt = (
        f"ประวัติการสนทนา:\n{history_text}\n\n"
        f"บริบท:\n{build_context(hits)}\n\nคำถาม: {question}{warning_text}"
    )
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=700,
        ),
    )
    return (resp.text or NOT_FOUND).strip()


# -------------------------------------------------------- 6) Chat UI
def show_sources(hits: list[dict]):
    with st.expander(f"📚 เอกสารอ้างอิง ({len(hits)} ชิ้น)"):
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}] {h['title']}**")
            st.caption(f"📄 {h['source']}  •  ความคล้าย {h['score']:.2f}")
            st.progress(max(0.0, min(1.0, h["score"])))
            st.caption(h["text"])


def show_welcome():
    st.markdown("""
    <div class="buddy-section-label">HOW IT WORKS</div>
    <div class="buddy-section-title">ถามได้อย่างมั่นใจ ตรวจสอบได้ทุกคำตอบ</div>
    <div class="buddy-section-sub">ระบบจะค้นหาเอกสารก่อนส่งบริบทให้ AI เรียบเรียง</div>
    <div class="buddy-grid">
      <div class="buddy-card"><div class="buddy-card-icon">🔎</div><strong>ค้นจากเอกสาร</strong><span>ค้นหาส่วนที่เกี่ยวข้องด้วย Semantic Search และ FAISS</span></div>
      <div class="buddy-card"><div class="buddy-card-icon">🧠</div><strong>ตอบด้วย Gemini</strong><span>เรียบเรียงคำตอบจากบริบทที่ค้นพบ ไม่เดาจากความรู้ภายนอก</span></div>
      <div class="buddy-card"><div class="buddy-card-icon">📚</div><strong>ตรวจสอบได้</strong><span>ทุกคำตอบมาพร้อมเอกสารอ้างอิงและคะแนนความคล้าย</span></div>
    </div>
    <div class="buddy-note">💡 <b>เริ่มต้นได้เลย:</b> พิมพ์คำถามด้านล่าง หรือเลือกคำถามตัวอย่างจากแถบด้านข้าง
    </div>
    """, unsafe_allow_html=True)


def show_document_library():
    """แสดงคลังความรู้บนหน้าเว็บ พร้อมอ่านและดาวน์โหลดไฟล์ต้นฉบับ"""
    paths = sorted(
        path for path in DATA_DIR.iterdir()
        if path.suffix.lower() in {".md", ".txt"}
    )
    labels = {}
    for path in paths:
        raw_text = path.read_text(encoding="utf-8")
        first_line = raw_text.splitlines()[0].lstrip("# ").strip() if raw_text else path.stem
        labels[path.name] = first_line

    with st.expander(f"📚 คลังเอกสารความรู้ ({len(paths)} หัวข้อ)"):
        selected_name = st.selectbox(
            "เลือกเอกสารที่ต้องการอ่าน",
            options=[path.name for path in paths],
            format_func=lambda name: labels[name],
            key="document_library_select",
        )
        selected_path = DATA_DIR / selected_name
        document_text = selected_path.read_text(encoding="utf-8")
        st.caption(f"📄 {selected_name} • {len(document_text):,} ตัวอักษร")
        preview, action = st.columns([4, 1])
        with preview:
            with st.container(height=360, border=True):
                st.markdown(document_text)
        with action:
            st.download_button(
                "⬇️ ดาวน์โหลด",
                data=document_text,
                file_name=selected_name,
                mime="text/markdown" if selected_path.suffix.lower() == ".md" else "text/plain",
                use_container_width=True,
                key="document_download",
            )
            st.info("เอกสารนี้เป็นส่วนหนึ่งของคลังที่ระบบใช้ค้นหาคำตอบ")


def export_history(messages: list[dict]) -> str:
    """สร้างบันทึกการสนทนาแบบ Markdown โดยไม่รวม API key หรือข้อมูลภายใน"""
    lines = ["# ประวัติการสนทนา NLP Buddy", ""]
    for message in messages:
        speaker = "👤 ผู้ใช้" if message["role"] == "user" else "🤖 NLP Buddy"
        lines.extend([f"## {speaker}", "", message["content"], ""])
        sources = list(dict.fromkeys(
            hit["source"] for hit in message.get("hits", [])
        ))
        if sources:
            lines.extend([f"**แหล่งอ้างอิง:** {', '.join(sources)}", ""])
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="NLP Buddy | RAG Assistant", page_icon="🤖",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="buddy-hero">
      <div class="buddy-hero-inner">
        <div>
          <div class="buddy-live"><span class="buddy-live-dot"></span> KNOWLEDGE ASSISTANT ONLINE</div>
          <div class="buddy-kicker">RETRIEVAL-AUGMENTED GENERATION</div>
          <h1>NLP Buddy <span style="color:#7dd3fc">✦</span></h1>
          <p>ผู้ช่วยเรียนรู้ด้านการประมวลผลภาษาธรรมชาติ ค้นจากคลังเอกสารภาษาไทย ตอบกระชับ และอ้างอิงกลับไปยังแหล่งข้อมูล</p>
          <div class="buddy-tags"><span class="buddy-tag">Sentence Embedding</span><span class="buddy-tag">FAISS</span><span class="buddy-tag">Gemini</span><span class="buddy-tag">Thai NLP</span></div>
        </div>
        <div class="rag-visual">
          <div class="rag-visual-title">RAG PIPELINE</div>
          <div class="rag-flow">
            <div class="rag-step"><div class="rag-step-icon">📄</div><div class="rag-step-text"><b>Knowledge Base</b><span>12 เอกสารความรู้</span></div></div>
            <div class="rag-step"><div class="rag-step-icon">🔎</div><div class="rag-step-text"><b>Semantic Search</b><span>Embedding + FAISS</span></div></div>
            <div class="rag-step"><div class="rag-step-icon">✦</div><div class="rag-step-text"><b>Grounded Answer</b><span>Gemini + Citations</span></div></div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    index, chunks = build_index()
    st.session_state.setdefault("messages", [])

    show_document_library()

    with st.sidebar:
        st.markdown("""<div class="buddy-brand"><div class="buddy-brand-title">🤖 NLP Buddy</div><div class="buddy-brand-sub">Your grounded NLP learning assistant</div></div><div class="buddy-status"><span class="buddy-dot"></span> ระบบพร้อมใช้งาน</div>""", unsafe_allow_html=True)
        st.subheader("📊 ภาพรวมระบบ")
        col1, col2 = st.columns(2)
        col1.metric("เอกสาร", len({c['source'] for c in chunks}))
        col2.metric("Chunks", len(chunks))
        st.write(f"Embedding: `{EMBED_MODEL}`")
        st.write(f"LLM: `{LLM_MODEL}` (Gemini)")
        st.divider()
        st.subheader("✨ คำถามแนะนำ")
        examples = ["WangchanBERTa คืออะไร",
                    "Self-Attention ทำงานอย่างไร",
                    "Chunk ควรมีขนาดเท่าไหร่ในระบบ RAG",
                    "ตัดคำภาษาไทยด้วยอะไรได้บ้าง"]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["pending"] = ex
        st.divider()
        st.subheader("🕘 ประวัติการสนทนา")
        user_messages = [
            message for message in st.session_state["messages"]
            if message["role"] == "user"
        ]
        if user_messages:
            st.caption(f"ทั้งหมด {len(user_messages)} คำถามในรอบนี้")
            for number, message in enumerate(user_messages[-5:][::-1], 1):
                question = message["content"]
                preview = question if len(question) <= 54 else question[:51] + "..."
                st.markdown(f"**{number}.** {preview}")
            st.download_button(
                "⬇️ ดาวน์โหลดประวัติ",
                data=export_history(st.session_state["messages"]),
                file_name="nlp_buddy_chat_history.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.caption("ยังไม่มีประวัติ ลองเริ่มถามคำถามได้เลย")
        if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    if not st.session_state["messages"]:
        show_welcome()

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("hits"):
                show_sources(m["hits"])

    question = st.chat_input("พิมพ์คำถามเกี่ยวกับ NLP...") or st.session_state.pop("pending", None)
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state["messages"]]
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาเอกสารและเรียบเรียงคำตอบ..."):
            unknown_terms = unknown_acronyms(question)
            hits = retrieve(retrieval_query(question, history))
            relevant = [h for h in hits if h["score"] >= MIN_SCORE]
            if not relevant:
                answer = f"{NOT_FOUND} ลองถามเกี่ยวกับหัวข้อ NLP ที่อยู่ในคลังความรู้ เช่น การตัดคำ Embedding Transformer หรือ RAG"
            else:
                try:
                    answer = ask_llm(question, relevant, history, unknown_terms)
                    if answer.startswith(NOT_FOUND):
                        relevant = []
                except Exception as e:  # เช่น ไม่ได้ตั้ง secret / เกินโควตา
                    answer = f"⚠️ เรียก LLM ไม่สำเร็จ: {type(e).__name__}"
                    relevant = []
        st.markdown(answer)
        if relevant:
            show_sources(relevant)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "hits": relevant})


if __name__ == "__main__":
    main()
