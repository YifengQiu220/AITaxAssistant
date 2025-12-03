# 修复 sqlite3 问题（必须在最开头）
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

import streamlit as st
import os
import sys
import tempfile

# --- 核心修正: 确保能找到同级目录下的 tax_brain ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 现在可以直接 import 了
from tax_brain import TaxOrchestrator, UserProfile

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
        # gpu=False 适合 Cloud 环境
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
                return "❌ PDF support not installed. Run: pip install PyPDF2"
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
        st.header("📋 User Profile")
        
        # ✅ 显示用户画像完整度
        user_profile = st.session_state.get('user_profile', UserProfile())
        
        # 计算完整度
        try:
            completeness = st.session_state.orchestrator.intake_agent.check_completeness(user_profile)
            completion_rate = completeness['completion_rate']
            
            # 进度条
            st.progress(completion_rate / 100)
            st.caption(f"Profile Completion: {completion_rate:.0f}%")
            
            # 详细字段状态
            st.divider()
            
            profile_fields = {
                "🌍 Citizenship": user_profile.citizenship_status,
                "🎓 Student Status": user_profile.student_status,
                "💼 Employment": user_profile.employment_details,
                "💰 Income": f"${user_profile.income:,}" if user_profile.income else None,
                "📍 State": user_profile.residency_state,
                "📝 Filing Experience": user_profile.tax_filing_experience,
            }
            
            for label, value in profile_fields.items():
                if value:
                    st.markdown(f"**{label}:** {value}")
                else:
                    st.markdown(f"**{label}:** ⬜ Not provided")
        
        except Exception as e:
            st.error(f"Error loading profile: {e}")
        
        st.divider()
        
        # ✅ 文件上传区域
        st.subheader("📎 Upload Documents")
        
        # 文档上传
        uploaded_doc = st.file_uploader(
            "Upload tax document (PDF/DOCX/TXT)",
            type=["pdf", "docx", "txt"],
            key="doc_uploader",
            help="Upload W-2, 1099, or other tax documents"
        )
        
        if uploaded_doc:
            with st.spinner("📄 Extracting text..."):
                extracted_text = extract_text_from_file(uploaded_doc)
                st.session_state.uploaded_doc_text = extracted_text
                st.session_state.uploaded_doc_name = uploaded_doc.name
            
            st.success(f"✅ Extracted from: {uploaded_doc.name}")
            
            with st.expander("📄 Preview", expanded=False):
                preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                st.text_area("Document content", preview_text, height=200, disabled=True)
            
            # ✅ 清除按钮
            if st.button("🗑️ Clear Document", key="clear_doc"):
                st.session_state.uploaded_doc_text = None
                st.session_state.uploaded_doc_name = None
                st.rerun()
        
        # 图片上传 (OCR)
        uploaded_img = st.file_uploader(
            "Upload W-2/1099 Image (OCR)",
            type=["png", "jpg", "jpeg"],
            key="img_uploader",
            help="Upload a photo of your tax form"
        )
        
        if uploaded_img:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(uploaded_img, caption="Uploaded", use_container_width=True)
            
            with col2:
                with st.spinner("🔍 Running OCR..."):
                    ocr_text = extract_text_from_image(uploaded_img)
                    st.session_state.uploaded_img_text = ocr_text
                    st.session_state.uploaded_img_name = uploaded_img.name
            
            st.success(f"✅ OCR completed: {uploaded_img.name}")
            
            with st.expander("🔍 OCR Result", expanded=False):
                st.text_area("Extracted text", ocr_text, height=200, disabled=True)
            
            # ✅ 清除按钮
            if st.button("🗑️ Clear Image", key="clear_img"):
                st.session_state.uploaded_img_text = None
                st.session_state.uploaded_img_name = None
                st.rerun()
        
        st.divider()
        
        # ==========================================
        # ✅ NEW: Checklist Display
        # ==========================================
        st.subheader("📋 Tax Filing Checklist")
        
        if st.session_state.get('checklist'):
            # Calculate overall completion
            all_sections = st.session_state.checklist
            if all_sections:
                total_completion = sum(s.get('completion', 0) for s in all_sections) / len(all_sections)
                st.progress(total_completion / 100)
                st.caption(f"Overall Progress: {total_completion:.0f}%")
                st.divider()
            
            # Display each section
            for section in st.session_state.checklist:
                heading = section.get("heading", "Unnamed Section")
                status = section.get("status", "pending")
                completion = section.get("completion", 0)
                details = section.get("details", [])
                
                # Section header with emoji
                status_emoji = "✅" if status == "done" else "⏳"
                
                with st.expander(f"{status_emoji} {heading} ({completion}%)", expanded=(completion < 100 and completion > 0)):
                    # Progress bar for this section
                    st.progress(completion / 100)
                    
                    # Display details
                    for detail in details:
                        item = detail.get("item", "")
                        d_status = detail.get("status", "pending")
                        d_emoji = "✅" if d_status == "done" else "⏳"
                        st.markdown(f"{d_emoji} {item}")
                    
                    st.caption(f"*{len([d for d in details if d.get('status') == 'done'])} of {len(details)} completed*")
        else:
            st.info("💡 Start chatting to see your personalized tax filing checklist!")
            st.caption("The checklist will automatically update as you provide information.")
        
        st.divider()
        
        # ✅ System Status
        st.subheader("🔧 System Status")
        
        # 显示 Agent 状态
        agent_status = {
            "Intake Agent": "✅ Ready",
            "RAG Agent": "✅ Ready (LangChain Chain)",
            "Tool Agent": "✅ Ready",
            "Checklist Agent": "✅ Ready (Progress Tracking)",
            "Orchestrator": "✅ Ready (LLM Decision)"
        }
        
        for agent, status in agent_status.items():
            st.caption(f"{status} - {agent}")
        
        # Debug 面板
        with st.expander("🧠 Debug Info", expanded=False):
            st.caption("**User Profile (JSON):**")
            if 'user_profile' in st.session_state:
                st.json(st.session_state.user_profile.dict(exclude_none=True))
            else:
                st.write("No data extracted yet.")
            
            st.caption("**Checklist (JSON):**")
            if 'checklist' in st.session_state and st.session_state.checklist:
                st.json(st.session_state.checklist)
            else:
                st.write("No checklist generated yet.")
            
            st.caption("**Session State Keys:**")
            st.write(list(st.session_state.keys()))

# ==========================================
# 主界面
# ==========================================
def main():
    st.title("AI Tax Assistant")
    st.caption("""Powered by Google Gemini 2.5 Pro + LangChain + RAG. Now, I can only assist NY state users.
    
If you need help with filling out tax form, please enter "hi" to start - there will be a simple questionnaire to help me better assist you. 

If you only need a simple answer, please ask directly to skip the user profile.""")
    # ✅ API Key 设置（修复版）
    if 'api_key' not in st.session_state:
        st.session_state.api_key = "AIzaSyD-NRi7pKPt-WalttQ9gPYpEFdhQv_TGZg"  # ← 替换成你的真实 Key
        try:
            # 从 secrets.toml 读取
            if st.secrets.get("GOOGLE_API_KEY"):
                 st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            pass

    # ✅ 初始化系统
    if 'orchestrator' not in st.session_state:
        with st.spinner("🔧 Initializing AI Tax Assistant..."):
            try:
                st.session_state.orchestrator = TaxOrchestrator(st.session_state.api_key)
                st.session_state.user_profile = UserProfile()
                st.session_state.messages = []
                st.session_state.checklist = []  # ← NEW: 初始化 checklist
                st.session_state.uploaded_doc_text = None
                st.session_state.uploaded_img_text = None
                st.session_state.uploaded_doc_name = None
                st.session_state.uploaded_img_name = None
                st.success("✅ System ready! All agents initialized.")
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
    
    # 渲染侧边栏
    render_sidebar()
    
    # ✅ 显示活跃的上传文档提示
    if st.session_state.get('uploaded_doc_text') or st.session_state.get('uploaded_img_text'):
        cols = st.columns([3, 1])
        with cols[0]:
            active_docs = []
            if st.session_state.get('uploaded_doc_name'):
                active_docs.append(f"📄 {st.session_state.uploaded_doc_name}")
            if st.session_state.get('uploaded_img_name'):
                active_docs.append(f"🖼️ {st.session_state.uploaded_img_name}")
            
            st.info(f"📎 Active documents: {', '.join(active_docs)}")
        
        with cols[1]:
            if st.button("🗑️ Clear All"):
                st.session_state.uploaded_doc_text = None
                st.session_state.uploaded_img_text = None
                st.session_state.uploaded_doc_name = None
                st.session_state.uploaded_img_name = None
                st.rerun()
    
    # ✅ 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # 显示 Agent 决策信息（如果有）
            if msg["role"] == "assistant" and "decision" in msg:
                with st.expander("🤖 Agent Decision Process", expanded=False):
                    st.caption(f"**Decision:** {msg['decision']}")
                    if "tools_used" in msg:
                        st.caption(f"**Tools Used:** {', '.join(msg['tools_used'])}")
    
    # ✅ 用户输入
    if prompt := st.chat_input("Ask me anything about your taxes..."):
        
        # 构建完整的上下文（包含上传的文档）
        context_parts = []
        tools_context = []
        
        if st.session_state.uploaded_doc_text:
            context_parts.append(f"[Document: {st.session_state.uploaded_doc_name}]\n{st.session_state.uploaded_doc_text[:2000]}")
            tools_context.append(f"📄 {st.session_state.uploaded_doc_name}")
        
        if st.session_state.uploaded_img_text:
            context_parts.append(f"[OCR from: {st.session_state.uploaded_img_name}]\n{st.session_state.uploaded_img_text}")
            tools_context.append(f"🖼️ {st.session_state.uploaded_img_name}")
        
        # 组合用户问题和上下文
        if context_parts:
            full_prompt = "\n\n".join(context_parts) + f"\n\nUser Question: {prompt}"
            display_prompt = f"{prompt}\n\n📎 *Using: {', '.join(tools_context)}*"
        else:
            full_prompt = prompt
            display_prompt = prompt
        
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": display_prompt})
        with st.chat_message("user"):
            st.markdown(display_prompt)
        
        # ✅ 后台提取用户信息（Intake Agent）
        with st.status("🔍 Analyzing your information...", expanded=False) as status:
            try:
                st.write("📋 Running Intake Agent...")
                new_data = st.session_state.orchestrator.run_intake(full_prompt)
                current_data = st.session_state.user_profile.dict()
                extracted_data = new_data.dict(exclude_none=True)
                
                # 显示新提取的字段
                if extracted_data:
                    st.write(f"✅ Extracted: {', '.join(extracted_data.keys())}")
                
                current_data.update(extracted_data)
                st.session_state.user_profile = UserProfile(**current_data)
                status.update(label="✅ Information extracted!", state="complete")
            except Exception as e:
                st.write(f"⚠️ Intake extraction warning: {e}")
                status.update(label="⚠️ Partial extraction", state="running")
        
        # ✅ 生成回答（Orchestrator + Agent Decision）
        with st.chat_message("assistant"):
            with st.status("🤖 AI is thinking...", expanded=True) as status:
                try:
                    st.write("🧠 Orchestrator analyzing query...")
                    st.write("🔄 Deciding which agents to use...")
                    
                    # 捕获 Agent 的决策输出
                    import io
                    import contextlib
                    
                    # 创建一个字符串缓冲区来捕获 print 输出
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        response = st.session_state.orchestrator.run_orchestrator(
                            full_prompt, 
                            st.session_state.user_profile
                        )
                    
                    # 获取捕获的输出
                    captured_output = f.getvalue()
                    
                    answer = response["output"]
                    
                    # 解析 Agent 决策
                    decision_info = {}
                    if "LLM Decision:" in captured_output:
                        decision_line = [line for line in captured_output.split('\n') if 'LLM Decision:' in line]
                        if decision_line:
                            decision_info['decision'] = decision_line[0].split('LLM Decision:')[1].strip()
                    
                    # 更新状态
                    if 'decision' in decision_info:
                        st.write(f"✅ Decision: {decision_info['decision']}")
                        
                        if decision_info['decision'] == "SEARCH":
                            st.write("🔍 Using: RAG Agent (searching IRS documents)")
                        elif decision_info['decision'] == "CALCULATE":
                            st.write("🧮 Using: Tool Agent (calculating taxes)")
                        elif decision_info['decision'] == "BOTH":
                            st.write("🔍 Using: RAG Agent + Tool Agent")
                        else:
                            st.write("💬 Using: Direct answer")
                    
                    status.update(label="✅ Answer generated!", state="complete")
                    
                    # 显示答案
                    st.markdown(answer)
                    
                    # 清除已使用的上传文档（避免重复使用）
                    if context_parts:
                        st.caption("📎 *Documents processed and cleared from context*")
                        st.session_state.uploaded_doc_text = None
                        st.session_state.uploaded_img_text = None
                        st.session_state.uploaded_doc_name = None
                        st.session_state.uploaded_img_name = None
                    
                except Exception as e:
                    answer = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(answer)
                    import traceback
                    st.code(traceback.format_exc())
                    status.update(label="❌ Error occurred", state="error")
                    decision_info = {"decision": "ERROR"}
        
        # 保存消息（包含决策信息）
        message_data = {"role": "assistant", "content": answer}
        if decision_info:
            message_data.update(decision_info)
        
        st.session_state.messages.append(message_data)
        
        # ==========================================
        # ✅ NEW: Generate Checklist After Each Turn
        # ==========================================
        with st.status("📋 Updating checklist...", expanded=False) as checklist_status:
            try:
                st.write("🔄 Checklist Agent analyzing conversation...")
                
                # Generate checklist
                checklist = st.session_state.orchestrator.generate_checklist(
                    conversation_history=st.session_state.messages,
                    user_profile=st.session_state.user_profile
                )
                
                st.session_state.checklist = checklist
                
                if checklist:
                    completed_sections = len([s for s in checklist if s.get('status') == 'done'])
                    total_sections = len(checklist)
                    st.write(f"✅ Checklist updated: {completed_sections}/{total_sections} sections completed")
                
                checklist_status.update(label="✅ Checklist updated!", state="complete")
                
            except Exception as e:
                st.write(f"⚠️ Checklist update failed: {e}")
                checklist_status.update(label="⚠️ Checklist update failed", state="error")
        
        st.rerun()


if __name__ == "__main__":
    main()