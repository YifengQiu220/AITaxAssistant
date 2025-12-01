import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import sys

# 修复 sqlite3 问题 (针对某些服务器环境)


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

# ==========================================
# 1. 设置配置 (和你的 VDB 代码保持一致)
# ==========================================
DB_DIRECTORY = "federal_tax_vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "federal_tax_documents"

# ==========================================
# 2. 定义数据结构 (Intake Agent 的输出)
# ==========================================
class UserProfile(BaseModel):
    """用户的基本税务画像"""
    name: Optional[str] = Field(description="User's name")
    filing_status: Optional[str] = Field(description="Filing status e.g., Single, Married Filing Jointly")
    income: Optional[int] = Field(description="Annual total income")
    w2_forms_count: Optional[int] = Field(description="Number of W-2 forms the user has")
    residency_state: Optional[str] = Field(description="State of residency e.g., CA, NY")

# ==========================================
# 3. 定义工具 (Tools)
# ==========================================

class TaxTools:
    def __init__(self):
        # 初始化 Embedding (必须和你存数据时用的一样!)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # 连接到你生成的 ChromaDB
        if os.path.exists(DB_DIRECTORY):
            self.db = Chroma(
                persist_directory=DB_DIRECTORY,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            print("✅ Successfully connected to existing ChromaDB")
        else:
            print("⚠️ Warning: ChromaDB directory not found. RAG will not work.")
            self.db = None

    def get_rag_tool(self):
        @tool
        def lookup_tax_knowledge(query: str, doc_type: str = "all"):
            """
            Useful for answering questions about tax rules, instructions, forms, or FAQs.
            query: The specific question (e.g. 'What is the standard deduction?')
            doc_type: Filter by 'instruction', 'form', 'faq', or 'all'.
            """
            if not self.db:
                return "Database not available."

            # 设置过滤器 (根据你之前存的 metadata)
            filter_dict = {}
            if doc_type != "all":
                filter_dict = {"doc_type": doc_type}
            
            # 执行搜索 (k=3)
            results = self.db.similarity_search(query, k=3, filter=filter_dict if filter_dict else None)
            
            # 格式化输出
            response = ""
            for doc in results:
                source = doc.metadata.get('source_file', 'Unknown')
                form = doc.metadata.get('form_number', 'N/A')
                response += f"Source: {source} (Form {form})\nContent: {doc.page_content}\n\n"
            
            return response
        return lookup_tax_knowledge

    def get_calculator_tool(self):
        @tool
        def tax_calculator(expression: str):
            """
            Useful for performing mathematical calculations. 
            Input should be a mathematical expression string (e.g. '50000 - 12000').
            """
            try:
                # 安全起见,只允许简单的数学运算
                allowed = set("0123456789+-*/(). ")
                if not all(c in allowed for c in expression):
                    return "Error: Only simple math allowed."
                return eval(expression)
            except Exception as e:
                return f"Error calculating: {str(e)}"
        return tax_calculator

# ==========================================
# 4. 核心大脑 (Orchestrator)
# ==========================================

class TaxOrchestrator:
    def __init__(self, api_key):
        # 使用 Google Gemini 2.0 Flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0
        )
        self.tax_tools = TaxTools()
        
        # 准备工具箱
        self.tools = [
            self.tax_tools.get_rag_tool(),
            self.tax_tools.get_calculator_tool()
        ]
        
        # 初始化 Agent
        self.agent_executor = self._build_agent()
        
        # 初始化结构化输出模型 (用于 Intake)
        self.extractor = self.llm.with_structured_output(UserProfile)

    def _build_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are the 'Orchestrator' for an AI Tax Assistant. 
            Your goal is to help the user complete their tax return by coordinating different tools.
            
            You have access to the following tools:
            1. lookup_tax_knowledge: Search the IRS database for rules (RAG).
            2. tax_calculator: Perform math.

            STRATEGY:
            - If the user provides personal info (e.g., "I earned 50k"), ACKNOWLEDGE it and say you've updated their profile.
            - If the user asks a law question, use 'lookup_tax_knowledge'.
            - If the user asks to calculate tax, use 'tax_calculator'.
            - Always be polite and professional.
            """),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 使用 create_tool_calling_agent 替代 create_openai_tools_agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def run_orchestrator(self, user_input):
        """主入口：处理用户输入并返回回答"""
        return self.agent_executor.invoke({"input": user_input})

    def run_intake(self, user_input):
        """专门用于提取用户信息的轻量级调用"""
        return self.extractor.invoke(user_input)