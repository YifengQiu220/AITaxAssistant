import os
import sys

# 修复 sqlite3 问题
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json

# ==========================================
# 配置
# ==========================================
DB_DIRECTORY = "federal_tax_vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "federal_tax_documents"

# ==========================================
# 数据结构
# ==========================================
class UserProfile(BaseModel):
    """用户的完整税务画像"""
    # Intake 调查问卷字段
    citizenship_status: Optional[str] = Field(default=None, description="US Citizen, Green Card, or Other")
    student_status: Optional[str] = Field(default=None, description="Full-time student, Part-time, or Not a student")
    employment_details: Optional[str] = Field(default=None, description="Employment type and details")
    tax_filing_experience: Optional[str] = Field(default=None, description="First time or experienced filer")
    residency_duration: Optional[str] = Field(default=None, description="How long lived in current state")
    income: Optional[int] = Field(default=None, description="Annual total income")
    residency_state: Optional[str] = Field(default=None, description="State of residency")
    
    # 其他可能的字段
    name: Optional[str] = Field(default=None, description="User's name")
    filing_status: Optional[str] = Field(default=None, description="Single, Married, etc.")
    w2_forms_count: Optional[int] = Field(default=None, description="Number of W-2 forms")

# ==========================================
# 1. Intake Agent - 问卷调查专家
# ==========================================
class IntakeAgent:
    """负责收集用户基本信息的调查问卷 Agent"""
    
    QUESTIONNAIRE = [
        "What is your citizenship status? (US Citizen / Green Card Holder / International Student / Other)",
        "Are you a student? (Full-time / Part-time / Not a student)",
        "What is your employment status? (On-campus job / Off-campus job / Self-employed / Unemployed)",
        "Have you filed taxes before? (First time / Filed before)",
        "How long have you lived in your current state?",
        "What was your total income last year?",
        "Which state do you currently live in?"
    ]
    
    def __init__(self, llm):
        self.llm = llm
        self.extractor = llm.with_structured_output(UserProfile)
    
    def get_questionnaire(self) -> str:
        """返回完整的问卷"""
        questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(self.QUESTIONNAIRE)])
        return f"""Welcome! To help you with your taxes, I need to ask a few questions:

{questions}

Please answer these questions, and I'll help you get started!"""
    
    def extract_info(self, user_input: str) -> UserProfile:
        """从用户回答中提取结构化信息"""
        try:
            return self.extractor.invoke(user_input)
        except Exception as e:
            print(f"⚠️ Intake extraction failed: {e}")
            return UserProfile()
    
    def check_completeness(self, profile: UserProfile) -> Dict[str, Any]:
        """检查问卷是否完成"""
        required_fields = [
            'citizenship_status', 'student_status', 'employment_details',
            'tax_filing_experience', 'income', 'residency_state'
        ]
        
        missing = []
        for field in required_fields:
            if getattr(profile, field) is None:
                missing.append(field)
        
        return {
            'complete': len(missing) == 0,
            'missing_fields': missing,
            'completion_rate': (len(required_fields) - len(missing)) / len(required_fields) * 100
        }

# ==========================================
# 2. RAG Agent - 知识检索专家
# ==========================================
class RAGAgent:
    """负责从 ChromaDB 检索税务知识"""
    
    def __init__(self, llm):
        self.llm = llm
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        if os.path.exists(DB_DIRECTORY):
            self.db = Chroma(
                persist_directory=DB_DIRECTORY,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            print("✅ RAG Agent: Connected to ChromaDB")
        else:
            print("⚠️ RAG Agent: ChromaDB not found")
            self.db = None
    
    def search(self, query: str, doc_type: str = "all", k: int = 3) -> str:
        """搜索相关税务文档"""
        if not self.db:
            return "Tax database is not available."
        
        try:
            filter_dict = {"doc_type": doc_type} if doc_type != "all" else None
            results = self.db.similarity_search(query, k=k, filter=filter_dict)
            
            if not results:
                return "No relevant information found in the tax database."
            
            response = "📚 **Information from IRS Documents:**\n\n"
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                form = doc.metadata.get('form_number', 'N/A')
                content = doc.page_content[:400]  # 限制长度
                response += f"**Source {i}:** {source} (Form {form})\n{content}...\n\n"
            
            return response
        except Exception as e:
            return f"Error searching database: {str(e)}"
    
    def answer_with_context(self, query: str, user_profile: UserProfile) -> str:
        """基于用户画像和检索结果生成回答"""
        # 先检索相关文档
        context = self.search(query)
        
        # 构建 prompt
        prompt = f"""You are a tax expert assistant. Answer the user's question based on the IRS documentation provided.

User Profile:
- Citizenship: {user_profile.citizenship_status or 'Unknown'}
- Student Status: {user_profile.student_status or 'Unknown'}
- Employment: {user_profile.employment_details or 'Unknown'}
- Income: ${user_profile.income or 'Unknown'}
- State: {user_profile.residency_state or 'Unknown'}

User Question: {query}

IRS Documentation:
{context}

Please provide a clear, helpful answer tailored to this user's situation."""

        response = self.llm.invoke(prompt)
        return response.content

# ==========================================
# 3. Tool Agent - 计算器专家
# ==========================================
class ToolAgent:
    """负责税务相关的计算"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def calculate(self, expression: str) -> str:
        """安全的数学计算"""
        try:
            # 只允许数字和基本运算符
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "❌ Error: Only basic math operations are allowed."
            
            result = eval(expression)
            return f"💰 Calculation Result: **{result:,.2f}**"
        except Exception as e:
            return f"❌ Calculation Error: {str(e)}"
    
    def calculate_tax(self, income: int, filing_status: str = "single") -> Dict[str, Any]:
        """计算联邦税（2024 税率表）"""
        # 2024 标准扣除额
        standard_deductions = {
            "single": 14600,
            "married_jointly": 29200,
            "married_separately": 14600,
            "head_of_household": 21900
        }
        
        # 2024 税率表 (Single)
        tax_brackets_single = [
            (11600, 0.10),
            (47150, 0.12),
            (100525, 0.22),
            (191950, 0.24),
            (243725, 0.32),
            (609350, 0.35),
            (float('inf'), 0.37)
        ]
        
        status = filing_status.lower().replace(" ", "_")
        deduction = standard_deductions.get(status, 14600)
        
        taxable_income = max(0, income - deduction)
        tax = 0
        prev_bracket = 0
        
        for bracket, rate in tax_brackets_single:
            if taxable_income <= bracket:
                tax += (taxable_income - prev_bracket) * rate
                break
            else:
                tax += (bracket - prev_bracket) * rate
                prev_bracket = bracket
        
        return {
            "gross_income": income,
            "standard_deduction": deduction,
            "taxable_income": taxable_income,
            "estimated_tax": round(tax, 2),
            "effective_rate": round((tax / income * 100), 2) if income > 0 else 0
        }

# ==========================================
# 4. Orchestrator Agent - 总指挥
# ==========================================
class OrchestratorAgent:
    """总协调器，决定调用哪个 Agent"""
    
    def __init__(self, llm, intake_agent, rag_agent, tool_agent):
        self.llm = llm
        self.intake = intake_agent
        self.rag = rag_agent
        self.tool = tool_agent
    
    def decide_agent(self, user_input: str, user_profile: UserProfile) -> str:
        """决定应该调用哪个 Agent"""
        
        # 检查用户画像完整度
        completeness = self.intake.check_completeness(user_profile)
        
        # 如果问卷未完成，优先使用 Intake Agent
        if not completeness['complete'] and any(keyword in user_input.lower() for keyword in ['hi', 'hello', 'start', 'help', 'begin']):
            return "intake"
        
        # 判断是否是计算问题
        calc_keywords = ['calculate', 'compute', 'how much', 'tax owed', 'estimate', 'math', 'minus', 'plus']
        if any(keyword in user_input.lower() for keyword in calc_keywords):
            return "tool"
        
        # 判断是否是知识查询
        knowledge_keywords = ['what is', 'how to', 'explain', 'form', 'deduction', 'credit', 'irs', 'rule', 'regulation']
        if any(keyword in user_input.lower() for keyword in knowledge_keywords):
            return "rag"
        
        # 默认使用 RAG Agent
        return "rag"
    
    def route(self, user_input: str, user_profile: UserProfile) -> str:
        """路由用户请求到相应的 Agent"""
        agent_type = self.decide_agent(user_input, user_profile)
        
        if agent_type == "intake":
            # 检查完整度
            completeness = self.intake.check_completeness(user_profile)
            if not completeness['complete']:
                return f"📋 {self.intake.get_questionnaire()}"
            else:
                return "✅ Great! I have all your information. How can I help you with your taxes today?"
        
        elif agent_type == "tool":
            # 提取数学表达式或使用用户收入计算
            if user_profile.income:
                result = self.tool.calculate_tax(
                    income=user_profile.income,
                    filing_status=user_profile.filing_status or "single"
                )
                return f"""💰 **Tax Calculation Results:**

- Gross Income: ${result['gross_income']:,}
- Standard Deduction: ${result['standard_deduction']:,}
- Taxable Income: ${result['taxable_income']:,}
- **Estimated Tax: ${result['estimated_tax']:,}**
- Effective Tax Rate: {result['effective_rate']}%

*This is an estimate based on 2024 federal tax rates.*"""
            else:
                return "To calculate your taxes, please tell me your annual income first."
        
        elif agent_type == "rag":
            return self.rag.answer_with_context(user_input, user_profile)
        
        return "I'm not sure how to help with that. Can you rephrase your question?"

# ==========================================
# 5. 主协调器（对外接口）
# ==========================================
class TaxOrchestrator:
    """主入口，管理所有 Agents"""
    
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0
        )
        
        # 初始化所有 Agents
        self.intake_agent = IntakeAgent(self.llm)
        self.rag_agent = RAGAgent(self.llm)
        self.tool_agent = ToolAgent(self.llm)
        self.orchestrator = OrchestratorAgent(
            self.llm,
            self.intake_agent,
            self.rag_agent,
            self.tool_agent
        )
    
    def run_orchestrator(self, user_input: str, user_profile: UserProfile = None) -> dict:
        """主入口：处理用户输入"""
        if user_profile is None:
            user_profile = UserProfile()
        
        response = self.orchestrator.route(user_input, user_profile)
        return {"output": response}
    
    def run_intake(self, user_input: str) -> UserProfile:
        """专门提取用户信息"""
        return self.intake_agent.extract_info(user_input)