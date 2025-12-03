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
    # Intake 调查问
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
    混合模式 Intake Agent
    ✅ 自动提取（你的） + 友好对话（队友的）
    """
    
    QUESTIONNAIRE = [
        "What is your citizenship status? (US Citizen / Green Card Holder / International Student / Other)",
        "Are you a student? (Full-time / Part-time / Not a student)",
        "What is your employment status? (On-campus job / Off-campus job / Self-employed / Multiple jobs)",
        "Have you filed US taxes before? (Yes / No)",
        "How long have you lived in your current state?",
        "What was your total income last year? (Approximate)",
        "Which state do you currently live in?"
    ]
    
    # ✅ 借鉴队友的友好 prompt
    CONVERSATIONAL_PROMPT = """I'm your AI tax assistant! 👋

I notice you need help with your taxes. To give you the best advice, I'd like to learn a bit about your situation.

You can either:
1. **Answer these quick questions:**
{questions}

2. **Or just tell me naturally**, like:
   - "I'm an international student on F-1 visa, working on-campus, earned $15k"
   - "I'm a working professional in California, made $60k last year"

I'll understand either way! 😊"""
    
    def __init__(self, llm):
        self.llm = llm
        self.extractor = llm.with_structured_output(UserProfile)
    
    def get_questionnaire(self) -> str:
        """返回友好对话式的问卷"""
        questions = "\n".join([f"   • {q}" for q in self.QUESTIONNAIRE])
        return self.CONVERSATIONAL_PROMPT.format(questions=questions)
    
    def extract_info(self, user_input: str) -> UserProfile:
        """自动提取（保持你的逻辑）"""
        try:
            return self.extractor.invoke(user_input)
        except Exception as e:
            print(f"⚠️ Intake extraction failed: {e}")
            return UserProfile()
    
    def check_completeness(self, profile: UserProfile) -> Dict[str, Any]:
        """检查完整度（保持你的逻辑）"""
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
    
    # ✅ 新增：智能追问（借鉴队友的对话式风格）
    def get_smart_followup(self, profile: UserProfile) -> str:
        """
        根据已有信息智能追问
        模仿队友的友好对话风格
        """
        completeness = self.check_completeness(profile)
        
        if completeness['complete']:
            return "✅ Perfect! I have everything I need. What would you like help with today?"
        
        missing = completeness['missing_fields']
        completion_rate = completeness['completion_rate']
        
        # 根据完整度调整语气
        if completion_rate < 30:
            # 刚开始：友好开放
            return """Thanks for that info! To help you better, could you share a bit more?

For example:
- Are you a student or working professional?
- What's your citizenship status?
- Approximate income last year?

Feel free to answer naturally! 😊"""
        
        elif completion_rate < 70:
            # 中间：具体追问
            friendly_questions = {
                'citizenship_status': "What's your citizenship status (US Citizen / Green Card / International Student)?",
                'student_status': "Are you currently a student?",
                'employment_details': "What's your employment situation?",
                'income': "What was your approximate income last year?",
                'residency_state': "Which state do you live in?",
            }
            
            next_question = friendly_questions.get(missing[0], f"Could you provide your {missing[0]}?")
            return f"Great! Just one more thing - {next_question}"
        
        else:
            # 接近完成：最后确认
            return f"Almost there! Just need to know: {', '.join(missing)}?"

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
- Student Status: {user_profile.student_status or 'Unknown'}
- Employment: {user_profile.employment_details or 'Unknown'}
- Income: ${user_profile.income or 'Unknown'}
- State: {user_profile.residency_state or 'Unknown'}

User Question: {user_input}

Decide what actions are needed:
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
                synthesis_prompt = f"""Based on this IRS information, answer the user's question in a friendly, helpful way.

User Profile:
- Citizenship: {user_profile.citizenship_status or 'Unknown'}
- Student Status: {user_profile.student_status or 'Unknown'}
- Income: ${user_profile.income or 'Unknown'}

User Question: {user_input}

IRS Information:
{context}

Provide a clear, helpful answer tailored to their situation. Be conversational and friendly! 😊"""
                response = self.llm.invoke(synthesis_prompt)
                return response.content
            
            elif action == "CALCULATE":
                if user_profile.income:
                    return self.tool.calculate_tax(user_profile.income)
                else:
                    return """I'd love to calculate your taxes! 💰

Could you tell me your approximate annual income? For example:
- "I earned $35,000 last year"
- "My income was around $50k"

I'll use this to estimate your federal tax."""
            
            elif action == "BOTH":
                # 先搜索
                context = self.rag.search(user_input)
                
                # 再计算
                tax_info = ""
                if user_profile.income:
                    tax_info = self.tool.calculate_tax(user_profile.income)
                
                # 综合
                synthesis_prompt = f"""Provide a comprehensive answer combining IRS documentation and tax calculation.

User Profile:
- Citizenship: {user_profile.citizenship_status or 'Unknown'}
- Student Status: {user_profile.student_status or 'Unknown'}
- Employment: {user_profile.employment_details or 'Unknown'}
- Income: ${user_profile.income or 'Unknown'}
- State: {user_profile.residency_state or 'Unknown'}

User Question: {user_input}

IRS Documentation:
{context}

Tax Calculation:
{tax_info if tax_info else "Income not provided - cannot calculate yet"}

Synthesize a clear, complete answer that:
1. Addresses their specific situation
2. Explains relevant IRS rules
3. Provides concrete numbers (if available)
4. Sounds friendly and helpful! 😊"""
                response = self.llm.invoke(synthesis_prompt)
                return response.content
            
            else:  # DIRECT
                return self.rag.answer_with_context(user_input, user_profile)
                
        except Exception as e:
            print(f"⚠️ LLM decision error: {e}")
            return self.rag.answer_with_context(user_input, user_profile)
    
    def route(self, user_input: str, user_profile: UserProfile) -> str:
        """
        主路由逻辑
        ✅ 集成新的友好对话式 Intake Agent
        """
        user_lower = user_input.lower().strip()
        
        # ✅ 改进 1: 简单问候 → 使用新的友好问卷
        if user_lower in ['hi', 'hello', 'hey', 'start', 'begin', 'help']:
            completeness = self.intake.check_completeness(user_profile)
            
            if not completeness['complete']:
                # 返回友好的对话式问卷
                return self.intake.get_questionnaire()
            else:
                # 全部信息已收集，使用智能追问
                return self.intake.get_smart_followup(user_profile)
        
        # ✅ 改进 2: 检查是否需要追问信息
        completeness = self.intake.check_completeness(user_profile)
        
        # 如果信息不完整且用户可能在补充信息
        if not completeness['complete'] and completeness['completion_rate'] < 70:
            # 检查用户输入是否像是在回答问卷
            answering_patterns = [
                'i am', "i'm", 'my', 'yes', 'no', 'student', 'citizen', 
                'green card', 'f-1', 'income', 'earned', 'state', 'live in'
            ]
            
            if any(pattern in user_lower for pattern in answering_patterns):
                # 用户可能在提供信息，先让 Intake Agent 提取
                # 然后返回智能追问
                followup = self.intake.get_smart_followup(user_profile)
                
                # 如果还需要更多信息，先追问
                if "almost there" not in followup.lower() and "perfect" not in followup.lower():
                    return followup
        
        # ✅ 改进 3: 其他查询 → LLM 决策（保持原有逻辑）
        print("🤖 Using LLM-enhanced decision making...")
        return self._llm_decide_and_act(user_input, user_profile)

# ==========================================
# 5. Checklist Agent - 进度追踪专家（普通类）
# ==========================================
class ChecklistAgent:
    """
    负责生成和维护税务清单
    ✅ 纯 LLM 调用，不需要 LangChain Agent
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_checklist(self, conversation_history: List[dict], user_profile: UserProfile) -> List[dict]:
        """
        基于对话历史和用户画像生成任务清单
        Returns: [{"heading": "...", "status": "done/pending", "completion": 0-100, "details": [...]}]
        """
        if not conversation_history:
            return []
        
        # 转换对话为文本
        convo_text = "\n".join([
            f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}" 
            for msg in conversation_history
        ])
        
        # System prompt
        system_prompt = """You are a CHECKLIST AGENT for US tax filing.

Based on the conversation and user profile, generate a hierarchical task checklist.

Return ONLY valid JSON in this EXACT format:
{
  "sections": [
    {
      "heading": "Collect W-2 forms",
      "status": "pending",
      "details": [
        {"item": "Collect W-2 from each employer for the tax year", "status": "done"},
        {"item": "Record wages, tips, other compensation (Box 1)", "status": "pending"},
        {"item": "Record federal income tax withheld (Box 2)", "status": "pending"}
      ]
    },
    {
      "heading": "Gather 1099 statements",
      "status": "pending",
      "details": [
        {"item": "Collect 1099-INT for interest income", "status": "pending"},
        {"item": "Collect 1099-DIV for dividend income", "status": "pending"}
      ]
    }
  ]
}

Rules:
- Use ACTION headings (e.g., "Collect W-2 forms", "Complete Form 1040-NR", "Gather 1099 statements")
- Each section should have 3-7 detailed sub-items
- Mark detail as "done" ONLY if user explicitly mentioned completing it in conversation
- Section status is "done" only if ALL its details are "done", otherwise "pending"
- Tailor sections to user profile:
  * International students: Form 1098-T, scholarship income, on-campus W-2
  * Working professionals: W-2, 1099 income, retirement contributions
  * Self-employed: 1099-NEC/1099-K, business expenses
- Provide 4-8 sections total
- Return ONLY the JSON, no extra text"""

        # User prompt
        user_context = f"""User Profile:
{json.dumps(user_profile.dict(exclude_none=True), indent=2)}

Conversation History:
{convo_text}

Generate the tax filing checklist as JSON:"""

        try:
            response = self.llm.invoke(f"{system_prompt}\n\n{user_context}")
            
            # Parse JSON from response
            import re
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                sections = data.get("sections", [])
                
                # Calculate completion percentage for each section
                for section in sections:
                    details = section.get("details", [])
                    if details:
                        done_count = sum(1 for d in details if d.get("status") == "done")
                        section["completion"] = int((done_count / len(details)) * 100)
                    else:
                        section["completion"] = 0
                    
                    # Auto-set section status
                    if section["completion"] == 100:
                        section["status"] = "done"
                    else:
                        section["status"] = "pending"
                
                return sections
                
        except json.JSONDecodeError as e:
            print(f"⚠️ Checklist JSON parsing failed: {e}")
        except Exception as e:
            print(f"⚠️ Checklist generation failed: {e}")
        
        return []


# ==========================================
# 6. 主协调器（对外接口）- 更新版
# ==========================================
class TaxOrchestrator:
    """主入口，管理所有 Agents"""
    
    def __init__(self, api_key):
        # ✅ 确保 API key 正确传递
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required. Please set it in your environment.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=api_key,
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
        
        self.checklist_agent = ChecklistAgent(self.llm)
        print("✅ Checklist Agent ready (普通类 - 进度追踪)")
        
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
    
    def generate_checklist(self, conversation_history: List[dict], user_profile: UserProfile = None) -> List[dict]:
        """
        生成税务清单（供 Streamlit 调用）
        
        Args:
            conversation_history: 对话历史 [{"role": "user", "content": "..."}]
            user_profile: 用户画像
        
        Returns:
            清单列表 [{"heading": "...", "status": "...", "completion": ..., "details": [...]}]
        """
        if user_profile is None:
            user_profile = UserProfile()
        
        return self.checklist_agent.generate_checklist(conversation_history, user_profile)

# ==========================================
# 5. Checklist Agent - 进度追踪专家（普通类）
# ==========================================
class ChecklistAgent:
    """
    负责生成和维护税务清单
    ✅ 纯 LLM 调用，不需要 LangChain Agent
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_checklist(self, conversation_history: List[dict], user_profile: UserProfile) -> List[dict]:
        """
        基于对话历史和用户画像生成任务清单
        """
        if not conversation_history:
            return []
        
        # 转换对话为文本
        convo_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in conversation_history
        ])
        
        # Prompt
        system_prompt = """You are a CHECKLIST AGENT for US tax filing.

Based on the conversation and user profile, generate a hierarchical task checklist.

Return ONLY valid JSON in this format:
{
  "sections": [
    {
      "heading": "Collect W-2 forms",
      "status": "pending",
      "completion": 50,
      "details": [
        {"item": "Collect W-2 from employer", "status": "done"},
        {"item": "Record Box 1 wages", "status": "pending"}
      ]
    }
  ]
}

Rules:
- Heading: Action-oriented (e.g., "Collect W-2 forms", "Complete Form 1040-NR")
- Status: "done" if ALL details are done, else "pending"
- Completion: 0-100 percentage
- Mark detail as "done" ONLY if user explicitly mentioned it
- Tailor to user profile (student → Form 1098-T, working → W-2, etc.)
- 4-8 sections total
"""

        user_prompt = f"""User Profile:
{json.dumps(user_profile.dict(exclude_none=True), indent=2)}

Conversation:
{convo_text}

Generate the checklist:"""

        try:
            response = self.llm.invoke(f"{system_prompt}\n\n{user_prompt}")
            
            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                sections = data.get("sections", [])
                
                # Normalize
                for section in sections:
                    if "completion" not in section:
                        # Calculate from details
                        details = section.get("details", [])
                        if details:
                            done_count = sum(1 for d in details if d.get("status") == "done")
                            section["completion"] = int((done_count / len(details)) * 100)
                        else:
                            section["completion"] = 0
                
                return sections
                
        except Exception as e:
            print(f"⚠️ Checklist generation failed: {e}")
        
        return []
    

# ==========================================
# 6. Visual Snippets - 表格映射教学（硬编码数据）
# ==========================================
VISUAL_SNIPPETS = {
    "w2_to_1040nr": [
        """
📋 W-2 → Form 1040-NR Mapping (Step 1/5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Focus: Box 1 (Wages, tips, other compensation)

W-2 Box 1: Wages, tips, other compensation
    ↓
Form 1040-NR Line 1a
    "Total amount from Form(s) W-2, box 1"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example: If Box 1 shows $45,000, enter $45,000 on Line 1a
        """,
        
        """
📋 W-2 → Form 1040-NR Mapping (Step 2/5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Focus: Box 2 (Federal income tax withheld)

W-2 Box 2: Federal income tax withheld
    ↓
Form 1040-NR Line 25a
    "Federal income tax withheld from Form(s) W-2"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example: If Box 2 shows $5,200, enter $5,200 on Line 25a
        """,
        
        """
📋 W-2 → Form 1040-NR Mapping (Step 3/5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Focus: Box 3-4 (Social Security wages and tax)

W-2 Box 3: Social Security wages
W-2 Box 4: Social Security tax withheld
    ↓
⚠️ Not entered directly on Form 1040-NR
    Used to verify Social Security records
    Check for excess withholding (Form 843)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """,
        
        """
📋 W-2 → Form 1040-NR Mapping (Step 4/5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Focus: Box 5-6 (Medicare wages and tax)

W-2 Box 5: Medicare wages and tips
W-2 Box 6: Medicare tax withheld
    ↓
⚠️ Not entered directly on Form 1040-NR
    Used to verify Medicare withholding amounts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """,
        
        """
📋 W-2 → Form 1040-NR Mapping (Step 5/5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Focus: Box 12, 14 (Other information)

W-2 Box 12: Codes (D, E, G, etc.)
    Retirement contributions (401k, etc.)
    → May affect Form 8880 or other forms

W-2 Box 14: "Other" information
    State tax, union dues, etc.
    → Relevant for state returns or recordkeeping
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ You've completed all W-2 → 1040-NR mappings!
        """
    ]
}