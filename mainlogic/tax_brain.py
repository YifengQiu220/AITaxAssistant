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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# ✅ 修复导入 - Tool 在这里
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Callable
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
# 1. Intake Agent - 问卷调查专家（普通类，不需要 LangChain）
# ==========================================
class IntakeAgent:
    """
    负责收集用户基本信息的调查问卷 Agent
    ✅ 固定流程，不需要 LangChain Agent
    """
    
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
# 2. RAG Agent - 知识检索专家（使用 LangChain Chain）
# ==========================================
class RAGAgent:
    """
    负责从 ChromaDB 检索税务知识
    ✅ 检索是固定流程，生成答案用 LangChain Chain
    """
    
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
        
        # ✅ 构建 LangChain Chain 用于生成答案
        self._build_qa_chain()
    
    def _build_qa_chain(self):
        """构建 RAG Chain（使用 LCEL）"""
        if not self.db:
            self.qa_chain = None
            return
        
        # 定义 prompt template
        template = """You are a tax expert assistant. Answer the user's question based on the IRS documentation provided.

User Profile:
- Citizenship: {citizenship_status}
- Student Status: {student_status}
- Employment: {employment_details}
- Income: ${income}
- State: {residency_state}

IRS Documentation:
{context}

User Question: {question}

Please provide a clear, helpful answer tailored to this user's situation. If the documentation doesn't contain relevant information, say so and provide general guidance."""

        prompt = ChatPromptTemplate.from_template(template)
        
        # ✅ 用 LCEL 构建 Chain
        def retrieve_and_format(inputs):
            """检索并格式化文档"""
            query = inputs["question"]
            docs = self.db.similarity_search(query, k=3)
            
            if not docs:
                return "No relevant information found in the tax database."
            
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                form = doc.metadata.get('form_number', 'N/A')
                content = doc.page_content[:400]
                formatted.append(f"Source {i} - {source} (Form {form}):\n{content}...")
            
            return "\n\n".join(formatted)
        
        self.qa_chain = (
            {
                "context": retrieve_and_format,
                "question": lambda x: x["question"],
                "citizenship_status": lambda x: x.get("citizenship_status", "Unknown"),
                "student_status": lambda x: x.get("student_status", "Unknown"),
                "employment_details": lambda x: x.get("employment_details", "Unknown"),
                "income": lambda x: x.get("income", "Unknown"),
                "residency_state": lambda x: x.get("residency_state", "Unknown"),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def search(self, query: str, doc_type: str = "all", k: int = 3) -> str:
        """
        简单搜索（不生成答案）
        用于：供 Orchestrator Agent 的 Tool 调用
        """
        if not self.db:
            return "Tax database is not available."
        
        try:
            filter_dict = {"doc_type": doc_type} if doc_type != "all" else None
            results = self.db.similarity_search(query, k=k, filter=filter_dict)
            
            if not results:
                return "No relevant information found in the tax database."
            
            response = "Information from IRS Documents:\n\n"
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                form = doc.metadata.get('form_number', 'N/A')
                content = doc.page_content[:300]
                response += f"Source {i} - {source} (Form {form}):\n{content}...\n\n"
            
            return response
        except Exception as e:
            return f"Error searching database: {str(e)}"
    
    def answer_with_context(self, query: str, user_profile: UserProfile) -> str:
        """
        基于用户画像和检索结果生成回答
        ✅ 使用 LangChain Chain
        """
        if not self.qa_chain:
            # Fallback: 如果没有数据库，直接用 LLM
            prompt = f"""You are a tax expert assistant.

User Profile:
- Citizenship: {user_profile.citizenship_status or 'Unknown'}
- Student Status: {user_profile.student_status or 'Unknown'}
- Employment: {user_profile.employment_details or 'Unknown'}
- Income: ${user_profile.income or 'Unknown'}
- State: {user_profile.residency_state or 'Unknown'}

User Question: {query}

Please provide helpful tax guidance based on general tax knowledge."""
            
            response = self.llm.invoke(prompt)
            return response.content
        
        try:
            # 使用 Chain
            chain_input = {
                "question": query,
                "citizenship_status": user_profile.citizenship_status or "Unknown",
                "student_status": user_profile.student_status or "Unknown",
                "employment_details": user_profile.employment_details or "Unknown",
                "income": user_profile.income or "Unknown",
                "residency_state": user_profile.residency_state or "Unknown",
            }
            
            response = self.qa_chain.invoke(chain_input)
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# ==========================================
# 3. Tool Agent - 计算器专家（普通类，不需要 LangChain）
# ==========================================
class ToolAgent:
    """
    负责税务相关的计算
    ✅ 纯数学逻辑，不需要 LangChain Agent
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def calculate(expression: str) -> str:
        """安全的数学计算"""
        try:
            # 只允许数字和基本运算符
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "❌ Error: Only basic math operations are allowed."
            
            result = eval(expression)
            return f"💰 Calculation Result: {result:,.2f}"
        except Exception as e:
            return f"❌ Calculation Error: {str(e)}"
    
    @staticmethod
    def calculate_tax(income: int, filing_status: str = "single") -> str:
        """
        计算联邦税（2024 税率表）
        返回格式化的字符串供 Agent 调用
        """
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
        
        effective_rate = round((tax / income * 100), 2) if income > 0 else 0
        
        return f"""Tax Calculation Results:
- Gross Income: ${income:,}
- Standard Deduction: ${deduction:,}
- Taxable Income: ${taxable_income:,}
- Estimated Tax: ${round(tax, 2):,}
- Effective Tax Rate: {effective_rate}%

This is an estimate based on 2024 federal tax rates."""

# ==========================================
# 4. Orchestrator Agent - 总指挥（使用 LangChain）
# ==========================================
class OrchestratorAgent:
    """
    总协调器
    ✅ 使用 LangChain 动态决策（简化版）
    """
    
    def __init__(self, llm, intake_agent, rag_agent, tool_agent):
        self.llm = llm
        self.intake = intake_agent
        self.rag = rag_agent
        self.tool = tool_agent
        
        # ✅ 定义可用的工具
        self.tools = {
            "search": Tool(
                name="search_tax_documents",
                func=self.rag.search,
                description="Search IRS tax documents for forms, deductions, and tax rules"
            ),
            "calculate": Tool(
                name="calculate_federal_tax",
                func=lambda income_str: self._calculate_tax_wrapper(income_str),
                description="Calculate federal income tax"
            )
        }
    
    def _calculate_tax_wrapper(self, income_str: str) -> str:
        """Wrapper for calculate_tax tool"""
        try:
            income = int(income_str.replace(",", "").replace("$", ""))
            return self.tool.calculate_tax(income)
        except ValueError:
            return "Error: Invalid income value."
    
    def _llm_decide_and_act(self, user_input: str, user_profile: UserProfile) -> str:
        """
        使用 LLM 决策需要什么操作，然后执行
        ✅ 简化版的 Agent 逻辑（不依赖复杂的 Agent 框架）
        """
        # 第一步：让 LLM 分析需要什么操作
        decision_prompt = f"""You are a tax assistant coordinator. Analyze what the user needs.

User Profile:
- Citizenship: {user_profile.citizenship_status or 'Unknown'}
- Income: ${user_profile.income or 'Unknown'}
- State: {user_profile.residency_state or 'Unknown'}

User Question: {user_input}

Decide what actions are needed (you can choose multiple):
1. SEARCH - Search IRS documents for information
2. CALCULATE - Calculate tax amount
3. BOTH - Need both search and calculation
4. DIRECT - Answer directly without tools

Respond with ONLY ONE WORD: SEARCH, CALCULATE, BOTH, or DIRECT"""

        try:
            decision = self.llm.invoke(decision_prompt)
            action = decision.content.strip().upper()
            print(f"🤖 LLM Decision: {action}")
            
            # 第二步：根据决策执行
            if action == "SEARCH":
                context = self.rag.search(user_input)
                synthesis_prompt = f"""Based on this IRS information, answer the user's question.

User Question: {user_input}

IRS Information:
{context}

Provide a clear, helpful answer."""
                response = self.llm.invoke(synthesis_prompt)
                return response.content
            
            elif action == "CALCULATE":
                if user_profile.income:
                    return self.tool.calculate_tax(user_profile.income)
                else:
                    return "I need your income to calculate taxes. What is your annual income?"
            
            elif action == "BOTH":
                # 先搜索
                context = self.rag.search(user_input)
                
                # 再计算
                tax_info = ""
                if user_profile.income:
                    tax_info = self.tool.calculate_tax(user_profile.income)
                
                # 综合
                synthesis_prompt = f"""Provide a comprehensive answer combining this information.

User Question: {user_input}

IRS Documentation:
{context}

Tax Calculation:
{tax_info if tax_info else "Income not provided"}

Synthesize a clear, complete answer."""
                response = self.llm.invoke(synthesis_prompt)
                return response.content
            
            else:  # DIRECT
                return self.rag.answer_with_context(user_input, user_profile)
                
        except Exception as e:
            print(f"⚠️ LLM decision error: {e}")
            return self.rag.answer_with_context(user_input, user_profile)
    
    def route(self, user_input: str, user_profile: UserProfile) -> str:
        """主路由逻辑"""
        user_lower = user_input.lower().strip()
        
        # 简单问候 → 规则路由
        if user_lower in ['hi', 'hello', 'hey', 'start', 'begin']:
            completeness = self.intake.check_completeness(user_profile)
            if not completeness['complete']:
                return f"📋 {self.intake.get_questionnaire()}"
            else:
                return "✅ Great! I have all your information. How can I help you with your taxes today?"
        
        # 其他查询 → LLM 决策
        print("🤖 Using LLM-enhanced decision making...")
        return self._llm_decide_and_act(user_input, user_profile)

class TaxOrchestrator:
    """主入口，管理所有 Agents"""
    
    def __init__(self, api_key):
        # ✅ 确保 API key 正确传递
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required. Please set it in your environment.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,  # 明确传递 API key
            temperature=0
        )
        
        # 初始化所有 Agents
        print("🚀 Initializing Tax Assistant System...")
        
        self.intake_agent = IntakeAgent(self.llm)
        print("✅ Intake Agent ready (普通类 - 固定流程)")
        
        self.rag_agent = RAGAgent(self.llm)
        print("✅ RAG Agent ready (LangChain Chain - 灵活生成)")
        
        self.tool_agent = ToolAgent()
        print("✅ Tool Agent ready (普通类 - 固定计算)")
        
        self.orchestrator = OrchestratorAgent(
            self.llm,
            self.intake_agent,
            self.rag_agent,
            self.tool_agent
        )
        print("✅ Orchestrator ready (LangChain 增强决策)")
        print("=" * 50)
    
    def run_orchestrator(self, user_input: str, user_profile: UserProfile = None) -> dict:
        """主入口：处理用户输入"""
        if user_profile is None:
            user_profile = UserProfile()
        
        response = self.orchestrator.route(user_input, user_profile)
        return {"output": response}
    
    def run_intake(self, user_input: str) -> UserProfile:
        """专门提取用户信息"""
        return self.intake_agent.extract_info(user_input)
