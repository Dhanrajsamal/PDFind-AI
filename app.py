import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

st.markdown("""
<style>

.stApp{
@keyframes bgMove{
    0%{
        background-position:0% 50%;
    }
    50%{
        background-position:100% 50%;
    }
    100%{
        background-position:0% 50%;
    }
}
    background:
    radial-gradient(circle at top left,#00E5FF22,transparent 35%),
    radial-gradient(circle at bottom right,#7B61FF22,transparent 35%),
    linear-gradient(135deg,#090B12,#111827,#0B1020);
    background-size:cover;
    animation:bgMove 12s ease infinite;
}
h1{
    color:#00E5FF;
    text-align:center;
}

h2,h3{
    color:white;
}

section[data-testid="stSidebar"]{
    background:#161B22;
}

div[data-testid="stFileUploader"]{
    border:2px dashed #00E5FF;
    border-radius:15px;
    padding:15px;
}

div.stButton > button{
    width:100%;
    height:48px;
    border-radius:14px;
    background:linear-gradient(90deg,#00E5FF,#7B61FF);
    color:white;
    border:none;
    font-weight:bold;
    transition:.3s;
}

div.stButton > button:hover{
    transform:scale(1.05);
    box-shadow:0 0 20px #00E5FF;
}

div[data-testid="stChatMessage"]{
    border-radius:15px;
    padding:12px;
    margin-bottom:10px;
}

/* Smooth Animation */
*{
    transition: all 0.3s ease;
}

/* Upload Box Hover */
div[data-testid="stFileUploader"]:hover{
    transform: scale(1.02);
    border:2px solid #00E5FF;
    box-shadow:0 0 25px rgba(0,229,255,.5);
}

/* Metric Cards Hover */
div[data-testid="stMetric"]{
    border-radius:15px;
    padding:10px;
    background:#1A1F2E;
}

div[data-testid="stMetric"]:hover{
    transform:translateY(-5px);
    box-shadow:0 0 20px rgba(0,229,255,.4);
}

/* AI Answer Box */
div[data-testid="stAlert"]{
    border-radius:15px;
}

/* Sidebar Animation */
section[data-testid="stSidebar"]{
    transition:0.3s;
}

section[data-testid="stSidebar"]:hover{
    box-shadow:0 0 20px rgba(0,229,255,.3);
}
/* Hero Section */
.hero{
    text-align:center;
    padding:30px 20px;
    margin:30px 10px;
    max-width:900px;
    margin:auto;

    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:25px;
    backdrop-filter:blur(15px);
}

.hero-title{
    font-size:72px;
    font-weight:800;
    color:#00E5FF;
    text-shadow:
        0 0 10px #00E5FF,
        0 0 20px #00E5FF,
        0 0 40px #00E5FF;
    animation: glow 2s infinite alternate;
}

.hero-subtitle{
    color:white;
    font-size:22px;
    margin-top:-10px;
}

.hero-powered{
    color:#9ca3af;
    font-size:16px;
    letter-spacing:1px;
}

@keyframes glow{
    from{
        text-shadow:
            0 0 10px #00E5FF,
            0 0 20px #00E5FF;
    }
    to{
        text-shadow:
            0 0 20px #00E5FF,
            0 0 40px #00E5FF,
            0 0 60px #00E5FF;
    }
}
@keyframes bgMove{

0%{
background-position:0% 50%;
}

50%{
background-position:100% 50%;
}

100%{
background-position:0% 50%;
}

}
div[data-testid="stVerticalBlock"]{
    background:rgba(255,255,255,0.03);
    backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px;
    padding:15px;
    transition:.3s;
}

div[data-testid="stVerticalBlock"]:hover{
    transform:translateY(-4px);
    box-shadow:0 0 25px rgba(0,229,255,.25);
}
div[data-testid="stFileUploader"]{
    animation:glowBox 2s infinite alternate;
}

@keyframes glowBox{
from{
box-shadow:0 0 10px #00E5FF55;
}
to{
box-shadow:0 0 25px #00E5FF;
}
}
.card{
    background:rgba(255,255,255,0.06);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);

    border:1px solid rgba(255,255,255,0.12);
    border-radius:22px;

    box-shadow:
        0 8px 32px rgba(0,0,0,.35),
        0 0 25px rgba(0,229,255,.08);

    transition:.35s;
}

.card:hover{
    transform:translateY(-8px) scale(1.02);

    box-shadow:
        0 15px 45px rgba(0,229,255,.25),
        0 0 40px rgba(123,97,255,.18);

    border:1px solid rgba(0,229,255,.35);
}
.stButton>button{
    width:100%;
    border-radius:14px;
    padding:12px 18px;

    background:linear-gradient(135deg,#00E5FF,#7B61FF);
    color:white;
    border:none;

    font-weight:700;
    transition:.35s;

    box-shadow:0 0 20px rgba(0,229,255,.25);
}

.stButton>button:hover{
    transform:translateY(-4px) scale(1.03);

    box-shadow:
        0 0 15px #00E5FF,
        0 0 35px #00E5FF,
        0 0 60px rgba(123,97,255,.45);

    filter:brightness(1.08);
}
.stApp::before{
    content:"";
    position:fixed;
    top:-200px;
    left:-200px;
    width:500px;
    height:500px;
    border-radius:50%;
    background:rgba(0,229,255,.12);
    filter:blur(120px);
    animation:float1 12s ease-in-out infinite;
    z-index:-1;
}

.stApp::after{
    content:"";
    position:fixed;
    right:-200px;
    bottom:-200px;
    width:500px;
    height:500px;
    border-radius:50%;
    background:rgba(123,97,255,.15);
    filter:blur(120px);
    animation:float2 14s ease-in-out infinite;
    z-index:-1;
}

@keyframes float1{
    0%,100%{transform:translate(0,0);}
    50%{transform:translate(120px,80px);}
}

@keyframes float2{
    0%,100%{transform:translate(0,0);}
    50%{transform:translate(-120px,-80px);}
}
.stApp::before{
    content:"";
    position:fixed;
    top:-200px;
    left:-200px;
    width:500px;
    height:500px;
    border-radius:50%;
    background:rgba(0,229,255,.12);
    filter:blur(120px);
    animation:float1 12s ease-in-out infinite;
    z-index:-1;
}

.stApp::after{
    content:"";
    position:fixed;
    right:-200px;
    bottom:-200px;
    width:500px;
    height:500px;
    border-radius:50%;
    background:rgba(123,97,255,.15);
    filter:blur(120px);
    animation:float2 14s ease-in-out infinite;
    z-index:-1;
}

@keyframes float1{
    0%,100%{transform:translate(0,0);}
    50%{transform:translate(120px,80px);}
}

@keyframes float2{
    0%,100%{transform:translate(0,0);}
    50%{transform:translate(-120px,-80px);}
}
/* User Message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
    background:rgba(0,229,255,.08);
    border:1px solid rgba(0,229,255,.25);
    border-radius:18px;
    padding:15px;
    margin-bottom:12px;
}

/* AI Message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){
    background:rgba(123,97,255,.08);
    border:1px solid rgba(123,97,255,.25);
    border-radius:18px;
    padding:15px;
    margin-bottom:12px;
}

/* Hover */
[data-testid="stChatMessage"]:hover{
    transform:translateY(-2px);
    transition:.3s;
    box-shadow:0 0 20px rgba(0,229,255,.18);
}
@keyframes fadeUp{
    from{
        opacity:0;
        transform:translateY(20px);
    }

    to{
        opacity:1;
        transform:translateY(0);
    }
}

[data-testid="stChatMessage"]{
    animation:fadeUp .4s ease;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="PDFMind AI | Dhanraj Samal",
    page_icon="🤖",
    layout="wide"
)
if "messages" not in st.session_state:
    st.session_state.messages = []
with st.sidebar:

    st.title("🤖 PDFMind AI")

    st.markdown("---")

    st.subheader("📂 Project")

    st.write("🧠 Local AI")
    st.write("📄 PDF Chatbot")
    st.write("⚡ Ollama")
    st.write("🗃️ ChromaDB")

    st.markdown("---")

    st.subheader("ℹ️ AI Model")

    st.success("Llama 3.2 : 3B")

    st.info("Embedding : nomic-embed-text")

    st.markdown("---")

    st.caption("Developed by Dhanraj Samal")

st.markdown("""
<div class="hero">

<h1 class="hero-title">
🤖 PDFMind AI
</h1>

<p class="hero-subtitle">
Chat with your PDFs using Local AI
</p>

<p class="hero-powered">
Powered by Llama 3.2 • ChromaDB • Ollama
<p style="color:#00E5FF;font-size:15px;">
Developed by Dhanraj Samal
</p>
</p>

</div>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🤖 AI Model", "Llama 3.2")

with col2:
    st.metric("🧠 Embeddings", "Nomic")

with col3:
    st.metric("📚 Vector DB", "Chroma")

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

    # Read PDF
    reader = PdfReader(uploaded_file)
    
    total_pages = len(reader.pages)

    st.info(f"📄 Total Pages: {total_pages}")

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Show first part of PDF
    st.subheader("📄 PDF Preview")
    st.text(text[:3000])

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    st.write(f"📝 Total Words: {len(text.split())}")
    st.write(f"📚 Total Chunks: {len(chunks)}")
    st.divider()

    st.write(f"📚 Total Chunks: {len(chunks)}")

    # Embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Vector Database
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    st.success("✅ Vector Database Ready!")

if st.button("📄 Generate PDF Summary"):

    with st.spinner("Generating Summary..."):

        llm = ChatOllama(
            model="llama3.2:3b"
        )

        prompt = f"""
Summarize this PDF in simple bullet points.

{text}
"""

        summary = llm.invoke(prompt)

    st.success("✅ Summary Generated")
    st.markdown("## 📄 PDF Summary")
    st.info(summary.content)

    # Question
    st.markdown("### ⚡ Quick Questions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 Summarize"):
        question = "Summarize this PDF"

with col2:
    if st.button("🎓 Education"):
        question = "What is the education?"

with col3:
    if st.button("💼 Experience"):
        question = "What is the work experience?"

col4, col5 = st.columns(2)

with col4:
    if st.button("🛠 Skills"):
        question = "What are the skills?"

with col5:
    if st.button("📞 Contact"):
        question = "What are the contact details?"

    question = st.chat_input("💬 Ask anything about your PDF....")

    if question:

        st.session_state.messages.append({
    "role": "user",
    "content": question
})

        

        with st.spinner("🤖 AI is Thinking..."):

            docs = vector_db.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            llm = ChatOllama(
                model="llama3.2:3b"
            )

            prompt = f"""
You are an AI assistant.

Answer ONLY from the context below.

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)

        st.success("✅ Answer Generated")

        st.markdown("## 🤖 AI Answer")
        st.caption("Generated using Llama 3.2 • Local AI • PDFMind AI")

        st.info(response.content)
        st.code(response.content, language=None)
        st.download_button(
    "📥 Download Answer",
    response.content,
    file_name="AI_Answer.txt",
    mime="text/plain"
)
        st.session_state.messages.append({
    "role": "assistant",
    "content": response.content
})
        st.markdown("---")

st.subheader("💬 Chat History")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        st.markdown("---")

st.caption("© 2026 PDFMind AI")

st.caption("Developed by Dhanraj Samal ❤️")