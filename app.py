import streamlit as st
import os
import requests
import base64
from io import BytesIO
from PIL import Image
from pypdf import PdfReader

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage 

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# ==========================================
# 1. SETUP & CONFIGURATION (Enhanced Glassmorphism)
# ==========================================
st.set_page_config(page_title="AI Clinical Assistant", layout="wide", page_icon="🩺")

# Custom CSS for Premium Glassmorphism and Unified Buttons
page_bg_img = """
<style>
/* Background Image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1557683316-973673baf926?q=80&w=2000&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Enhanced Glassmorphism Container */
.glass-panel {
    background: rgba(255, 255, 255, 0.08); /* Slightly more transparent background */
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); /* Deeper shadow for better separation */
    backdrop-filter: blur(24px); /* Significantly increased blur */
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.2); /* Softer highlight border */
    padding: 25px;
    margin-bottom: 20px;
    color: #ffffff;
}

/* Unified & Premium Button Styling */
.stButton > button {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease-in-out !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    font-weight: 600 !important;
}

/* Button Hover Effect */
.stButton > button:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
}

/* Text overrides for readability on dark background */
h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown {
    color: #f8f9fa !important;
}

/* Make standard streamlit containers transparent so glass shows */
[data-testid="stVerticalBlock"] {
    background-color: transparent;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Initialize Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# 2. SIDEBAR (API Key & Chat Controls)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color: white;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    st.markdown("<h3 style='color: white;'>🗂️ History Controls</h3>", unsafe_allow_html=True)
    if st.button("🗑️ Delete Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("History cleared!")

# Main Title
st.markdown("<div class='glass-panel'><h1>🩺 Multimodal Clinical Decision Support System</h1><p>Prototype: Demonstrates RAG, Vision-Language Models, and API Integration. NOT FOR REAL MEDICAL USE.</p></div>", unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def fetch_fda_drug_info(drug_name):
    try:
        url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug_name}&limit=1"
        response = requests.get(url).json()
        if 'results' in response:
            reactions = response['results'][0]['patient']['reaction']
            return [reaction['reactionmeddrapt'] for reaction in reactions[:3]]
        return ["No specific data found."]
    except Exception:
        return ["API Error or Drug not found."]

# ==========================================
# 4. RAG PIPELINE
# ==========================================
@st.cache_resource
def initialize_rag_database(_api_key):
    mock_medical_guidelines = [
        "Elevated WBC count (above 11,000/mcL) combined with fever may indicate a systemic infection or pneumonia.",
        "Cardiomegaly on a chest X-ray is indicated by a cardiothoracic ratio > 0.5 and may suggest heart failure.",
        "Standard treatment for community-acquired pneumonia often includes Amoxicillin or Azithromycin.",
        "Opaque infiltrates in the lower lung lobes on an X-ray are classic signs of bacterial pneumonia.",
        "Hyperintense lesions on T2-weighted Brain MRIs can indicate demyelinating diseases or tumors.",
        "A midline shift or mass effect on a Brain MRI suggests elevated intracranial pressure, often managed with corticosteroids like Dexamethasone.",
        "Ischemic strokes present as areas of restricted diffusion on an MRI brain scan.",
        "Cortical disruptions or radiolucent lines on a skeletal X-ray typically indicate a bone fracture.",
        "Standard initial management for an uncomplicated fracture includes immobilization and pain management with NSAIDs like Ibuprofen."
    ]
    
    docs = [Document(page_content=text) for text in mock_medical_guidelines]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()

# ==========================================
# 5. MAIN UI & EXECUTION
# ==========================================
if not api_key:
    st.warning("Please enter your API key in the sidebar to begin.")
    st.stop()

try:
    retriever = initialize_rag_database(api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

# --- INPUT SECTION ---
st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Text/PDF Report")
    uploaded_pdf = st.file_uploader("Blood Test or Clinical Notes", type=["pdf"])
    
with col2:
    st.subheader("2. Upload Medical Scan")
    uploaded_image = st.file_uploader("X-Ray, MRI, CT", type=["png", "jpg", "jpeg"])

# Removed type="primary" so this button matches the sidebar delete button perfectly
analyze_button = st.button("Generate Diagnostic Insights", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- ANALYSIS & HISTORY SECTION ---
if analyze_button:
    if not uploaded_pdf and not uploaded_image:
        st.error("Please upload at least one file.")
    else:
        with st.spinner("Analyzing patient data..."):
            report_text = ""
            image_analysis = ""
            
            if uploaded_pdf:
                report_text = extract_text_from_pdf(uploaded_pdf)
            
            if uploaded_image:
                image_bytes = uploaded_image.getvalue()
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                mime_type = uploaded_image.type
                image_data_uri = f"data:{mime_type};base64,{base64_image}"
                
                vision_prompt = "You are a radiologist. First, identify the type of scan (e.g., Chest X-Ray, Brain MRI). Then, analyze it and note any visible abnormalities. Be concise."
                vision_message = HumanMessage(
                    content=[
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                )
                vision_response = llm.invoke([vision_message])
                image_analysis = vision_response.content
            
            prompt_template = ChatPromptTemplate.from_template("""
            You are a Clinical Decision Support AI.
            Task: {input}
            
            Based on the following medical guidelines from our database:
            <context>
            {context}
            </context>
            
            Patient Report Text: {report}
            Image Scan Analysis: {scan}
            
            INSTRUCTIONS:
            1. Provide a summary of the potential findings based on the Patient Report and Image Scan Analysis.
            2. Attempt to align these findings with the retrieved guidelines in the <context>.
            3. CRITICAL: If the retrieved guidelines are completely irrelevant to the findings (e.g., matching a Brain MRI to Pneumonia guidelines), DO NOT force an alignment. Instead, explicitly state: "No relevant clinical guidelines found in the current database for this specific scan/report."
            4. Suggest a potential medication ONLY if it is supported by the relevant guidelines or standard general practice for the finding.
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            response = retrieval_chain.invoke({
                "input": "Analyze the patient data.",
                "report": report_text if report_text else "No report provided.",
                "scan": image_analysis if image_analysis else "No scan provided."
            })
            
            # --- FDA Check ---
            demo_drugs = ["Amoxicillin", "Azithromycin", "Lisinopril", "Dexamethasone", "Ibuprofen"]
            found_drug = None
            fda_info = ""
            for drug in demo_drugs:
                if drug.lower() in response["answer"].lower():
                    found_drug = drug
                    break
            
            if found_drug:
                side_effects = fetch_fda_drug_info(found_drug)
                fda_info = f"\n\n**💊 FDA Drug Identified:** {found_drug}\n**Adverse Reactions (openFDA):** {', '.join(side_effects)}"
            
            final_output = response["answer"] + fda_info
            
            # Save to history
            st.session_state.chat_history.insert(0, {
                "files": f"{uploaded_pdf.name if uploaded_pdf else ''} | {uploaded_image.name if uploaded_image else ''}",
                "result": final_output
            })

# --- RENDER CHAT HISTORY ---
if st.session_state.chat_history:
    st.markdown("<h2 style='color: white;'>🕰️ Past Analyses</h2>", unsafe_allow_html=True)
    for entry in st.session_state.chat_history:
        st.markdown(f"""
        <div class='glass-panel'>
            <strong>📁 Files Analyzed:</strong> {entry['files']}<br><br>
            <strong>🧠 AI Assessment:</strong><br>
            {entry['result'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)