import streamlit as st
import os
from mainlogic.tax_brain import TaxOrchestrator, UserProfile
import tempfile

# OCR 相关
try:
    import easyocr
    from PIL import Image
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# 文档处理相关
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="AI Tax Assistant", 
    page_icon="📝", 
    layout="wide"
)

# ==========================================
# OCR 模型加载（缓存）
# ==========================================
@st.cache_resource
def load_ocr():
    if OCR_AVAILABLE:
        return easyocr.Reader(["en"], gpu=False)
    return None

# ==========================================
# 文档文本提取函数
# ==========================================
def extract_text_from_file(uploaded_file):
    """从上传的文件中提取文本"""
    file_type = uploaded_file.type
    text = ""
    
    try:
        if file_type == "text/plain":
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            
        elif file_type == "application/pdf":
            if not PDF_AVAILABLE:
                return "❌ PDF support not installed. Run: pip install pypdf2"
            reader = PdfReader(uploaded_file)
            text_chunks = []
            for page in reader.pages[:10]:  # 限制前10页
                text_chunks.append(page.extract_text() or "")
            text = "\n".join(text_chunks)
            
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            if not DOCX_AVAILABLE:
                return "❌ DOCX support not installed. Run: pip install python-docx"
            doc = Document(uploaded_file)
            paragraphs = [p.text for p in doc.paragraphs]
            text = "\n".join(paragraphs)
            
    except Exception as e:
        text = f"❌ Error extracting text: {str(e)}"
    
    return text

# ==========================================
# OCR 文本提取函数
# ==========================================
def extract_text_from_image(uploaded_image):
    """从上传的图片中提取文本（OCR）"""
    if not OCR_AVAILABLE:
        return "❌ OCR not installed. Run: pip install easyocr pillow"
    
    try:
        ocr_reader = load_ocr()
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        results = ocr_reader.readtext(image_np)
        extracted_text = "\n".join([res[1] for res in results])
        return extracted_text
    except Exception as e:
        return f"❌ OCR Error: {str(e)}"

# ==========================================
# 侧边栏
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.header("📋 Filing Progress")
        
        # 进度追踪
        user_profile = st.session_state.get('user_profile', UserProfile())
        steps = {
            "Personal Info": user_profile.name is not None,
            "Income Data": user_profile.income is not None,
            "Filing Status": user_profile.filing_status is not None,
            "Residency": user_profile.residency_state is not None,
        }
        
        for step, done in steps.items():
            icon = "✅" if done else "⬜"
            st.markdown(f"{icon} {step}")
        
        st.divider()
        
        # 文件上传区域
        st.subheader("📎 Upload Documents")
        
        # 文档上传
        uploaded_doc = st.file_uploader(
            "Upload tax document (PDF/DOCX/TXT)",
            type=["pdf", "docx", "txt"],
            key="doc_uploader"
        )
        
        if uploaded_doc:
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_file(uploaded_doc)
                st.session_state.uploaded_doc_text = extracted_text
            
            with st.expander("📄 Preview", expanded=False):
                st.text(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
        
        # 图片上传
        uploaded_img = st.file_uploader(
            "Upload W-2/1099 Image (OCR)",
            type=["png", "jpg", "jpeg"],
            key="img_uploader"
        )
        
        if uploaded_img:
            st.image(uploaded_img, caption="Uploaded", use_container_width=True)
            with st.spinner("Running OCR..."):
                ocr_text = extract_text_from_image(uploaded_img)
                st.session_state.uploaded_img_text = ocr_text
            
            with st.expander("🔍 OCR Result", expanded=False):
                st.text(ocr_text)
        
        st.divider()
        
        # Debug 面板
        with st.expander("🧠 Memory (Debug)", expanded=False):
            if 'user_profile' in st.session_state:
                st.json(st.session_state.user_profile.dict(exclude_none=True))
            else:
                st.write("No data extracted yet.")

# ==========================================
# 主界面
# ==========================================
def main():
    st.title("🤖 AI Tax Assistant")
    st.caption("Powered by Google Gemini 2.0 Flash + RAG")
    
    # API Key 输入
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv("GOOGLE_API_KEY")
    
    if not st.session_state.api_key:
        st.warning("⚠️ Please configure your Google API Key")
        key = st.text_input("Enter Google API Key:", type="password")
        if key:
            st.session_state.api_key = key
            st.rerun()
        return
    
    # 初始化系统
    if 'orchestrator' not in st.session_state:
        with st.spinner("🔧 Initializing AI Tax Assistant..."):
            try:
                st.session_state.orchestrator = TaxOrchestrator(st.session_state.api_key)
                st.session_state.user_profile = UserProfile()
                st.session_state.messages = []
                st.session_state.uploaded_doc_text = None
                st.session_state.uploaded_img_text = None
                st.success("✅ System ready!")
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                return
    
    # 渲染侧边栏
    render_sidebar()
    
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 用户输入
    if prompt := st.chat_input("Ask me anything about your taxes..."):
        
        # 构建完整的上下文（包含上传的文档）
        context_parts = []
        
        if st.session_state.uploaded_doc_text:
            context_parts.append(f"**Uploaded Document:**\n{st.session_state.uploaded_doc_text[:1000]}")
        
        if st.session_state.uploaded_img_text:
            context_parts.append(f"**OCR from Image:**\n{st.session_state.uploaded_img_text}")
        
        # 组合用户问题和上下文
        if context_parts:
            full_prompt = "\n\n".join(context_parts) + f"\n\n**User Question:** {prompt}"
        else:
            full_prompt = prompt
        
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 后台提取用户信息（Intake Agent）
        try:
            new_data = st.session_state.orchestrator.run_intake(full_prompt)
            current_data = st.session_state.user_profile.dict()
            extracted_data = new_data.dict(exclude_none=True)
            current_data.update(extracted_data)
            st.session_state.user_profile = UserProfile(**current_data)
        except Exception as e:
            print(f"⚠️ Intake extraction failed: {e}")
        
        # 生成回答（Orchestrator + RAG）
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking... (Checking IRS documents)"):
                try:
                    response = st.session_state.orchestrator.run_orchestrator(full_prompt)
                    answer = response["output"]
                    st.markdown(answer)
                    
                    # 清除已使用的上传文档（避免重复使用）
                    st.session_state.uploaded_doc_text = None
                    st.session_state.uploaded_img_text = None
                    
                except Exception as e:
                    answer = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    main()