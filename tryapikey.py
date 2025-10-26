import os
import json
import time
import threading
import re
import pandas as pd
import numpy as np # <<< ADDED IMPORT FOR NUMPY
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import requests


# =========================
# Environment & Configuration
# =========================
load_dotenv()  # Load variables from .env file

# --- Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "LOCAL") # "GEMINI", "OPENAI", or "LOCAL"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "C:/Users/perna/Desktop/NOSTRADECIUS/openhermes-2-mistral-7b.Q4_K_M.gguf")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "models_cache")
FILE_MANIFEST_PATH = "file_manifest.json"


# =========================
# Flask app & Globals
# =========================
app = Flask(__name__)
CORS(app)

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None)

llm_client = None
embedding_model = None
tools = {}
file_manifest = []
conversation_memory_store = {}
llm_lock = threading.Lock()


# =========================
# Prompts
# =========================
ROUTER_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é agir como um roteador de IA. Com base na pergunta mais recente do usuário e no histórico da conversa, selecione a melhor ferramenta da lista para respondê-la. Responda APENAS com o nome da ferramenta (`tool_name`).<|im_end|>
<|im_start|>user
--- Histórico da Conversa ---
{chat_history}
-----------------------------
--- Pergunta Mais Recente ---
"{question}"
-----------------------------
--- Ferramentas Disponíveis ---
{tools_description}
--------------------------------<|im_end|>
<|im_start|>assistant
"""
)

GENERAL_QA_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Você é o Nostradecius, um assistente de IA prestativo e factual. Sua principal função é responder usando o CONTEXTO.

REGRAS ESTRITAS:
1. Responda em Português do Brasil.
2. **Prioridade Máxima:** Se o CONTEXTO tiver informações, baseie sua resposta APENAS nele.
3. **Exceção (Contexto Vazio):** Se o CONTEXTO estiver vazio:
    a. Se a pergunta for uma saudação (como 'oi', 'opa', 'tudo bem?'), responda de forma educada e prestativa.
    b. Para qualquer outra pergunta, responda EXATAMENTE: 'Não encontrei dados relevantes sobre isso.'
4. Não use conhecimento externo ou faça suposições.<|im_end|>
<|im_start|>user
CONTEXTO:
{context}

PERGUNTA DO USUÁRIO: {input}<|im_end|>
<|im_start|>assistant
RESPOSTA:
"""
)


CODE_GENERATOR_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é agir como um especialista em pandas Python. Dada uma pergunta do usuário e uma lista de colunas, seu objetivo é gerar uma ÚNICA linha de código Python que filtra e seleciona dados de um DataFrame chamado `df`.

REGRAS CRÍTICAS E ABSOLUTAS:
1.  **Gere APENAS uma linha de código Python.** Não use `print()`, não use comentários, não use markdown como ```python. A saída deve ser apenas o código.
2.  **NÃO INVENTE VALORES:** Use apenas os valores exatos da pergunta do usuário. Se o usuário pedir por "Decio", procure por "Decio". **NUNCA adicione prefixos como "Analista"** (ex: "Analista Decio") a menos que o usuário tenha digitado exatamente isso.
3.  **DATAS (`datetime64[ns]`):** Para filtrar datas, **SEMPRE** use o acessador `.dt`. Exemplo: `df['Data Criação'].dt.year == 2025`.
4.  **TEXTO (`object`):** Para colunas de texto, **SEMPRE** use `str.contains('valor', case=False, na=False)`.
5.  **SELEÇÃO DE COLUNAS:** Após a filtragem, selecione APENAS as colunas que o usuário pediu para ver. Se ele não pedir colunas específicas, mostre colunas relevantes.

EXEMPLO 1 (Nome de Analista):
Pergunta: "Quantos projetos o analista Decio tem com status On Hold?"
Colunas e Tipos:
Nome Projeto      object
Nome Analista     object
Status            object
Data Criação      datetime64[ns]
Código Gerado:
df_result = df[(df['Nome Analista'].str.contains('Decio', case=False, na=False)) & (df['Status'].str.contains('On Hold', case=False, na=False))]

EXEMPLO 2 (Contagem):
Pergunta: "Quantos projetos o analista Decio tem?"
Colunas e Tipos:
Nome Projeto      object
Nome Analista     object
Código Gerado:
df_result = df[df['Nome Analista'].str.contains('Decio', case=False, na=False)].shape[0]

EXEMPLO 3 (Filtro de Data):
Pergunta: "me mostre os projetos criados em 2024"
Colunas e Tipos:
Nome Projeto      object
Data Criação      datetime64[ns]
Código Gerado:
df_result = df[df['Data Criação'].dt.year == 2024]
<|im_end|>
<|im_start|>user
Pergunta: "{question}"
Colunas e Tipos:
{columns_and_types}

Gere o código Python de uma linha.<|im_end|>
<|im_start|>assistant
"""
)

CODE_REPAIR_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é agir como um depurador de código Python para pandas. Um código anterior falhou ao tentar responder a pergunta de um usuário. Seu objetivo é gerar uma nova e corrigida linha de código Python.

INFORMAÇÕES DISPONÍVEIS:
1.  **Pergunta Original do Usuário:** A intenção final do usuário.
2.  **Código com Erro:** O código que gerou um erro ou não retornou resultados.
3.  **Feedback do Erro:** A mensagem de erro ou a indicação de "Nenhum resultado encontrado".
4.  **Amostra de Dados (`df.head()`):** As primeiras linhas do dataframe, mostrando os nomes exatos das colunas e o formato dos dados.

REGRAS CRÍTICAS PARA A CORREÇÃO:
1.  **Observe a Amostra de Dados:** Compare o `Código com Erro` com a `Amostra de Dados`. Preste atenção especial a:
    * Nomes de colunas (ex: `Nome Analista` vs `Analista`).
    * Valores de texto (ex: o código procurou por "ongoing", mas os dados contêm "On Going"; o código procurou por "Analista JUSCEDIR", mas os dados contêm "JUSCEDIR JUNIOR (DHL)").
    * Uso de crases (`) ou markdown, que são inválidos. O código deve ser Python puro.
2.  **Gere APENAS uma linha de código corrigido.** Não explique, não comente, apenas o código.

EXEMPLO DE CORREÇÃO:
- **Pergunta Original:** "Liste projetos do Analista Decio"
- **Código com Erro:** `df_result = df[df['Nome Analista'].str.contains('Analista Decio', case=False, na=False)]`
- **Feedback do Erro:** "Nenhum resultado encontrado"
- **Amostra de Dados:**
      Nome Analista
  0   DECIO OLIVEIRA (DHL)
  1   VINCENT PERNARH (DHL)
- **Código Corrigido Gerado:**
  `df_result = df[df['Nome Analista'].str.contains('Decio', case=False, na=False)]`
<|im_end|>
<|im_start|>user
---
**Pergunta Original do Usuário:**
"{question}"
---
**Código com Erro:**
`{failed_code}`
---
**Feedback do Erro:**
"{error_feedback}"
---
**Amostra de Dados (`df.head()`):**
{df_head}
---
Gere a linha de código Python corrigida.<|im_end|>
<|im_start|>assistant
"""
)


SUMMARIZER_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é atuar como um assistente de IA que resume os dados encontrados de forma clara e natural em Português.

REGRAS CRÍTICAS:
1.  Baseie sua resposta ESTRITAMENTE nos dados fornecidos: `PERGUNTA ORIGINAL DO USUÁRIO`, `CONTAGEM TOTAL REAL`, e `AMOSTRA DE DADOS`. Não invente informações.
2.  Comece com uma única frase que resume o que foi encontrado, conectando-se diretamente à pergunta do usuário.
3.  Após a frase de resumo, introduza a lista de dados de forma precisa.
4.  Use Markdown para formatar a resposta. Destaque os principais termos da pergunta do usuário em negrito (`**palavra**`).
<|im_end|>
<|im_start|>user
PERGUNTA ORIGINAL DO USUÁRIO:
"{question}"

CONTAGEM TOTAL REAL: {total_count}

AMOSTRA DE DADOS EXIBIDA (pode ser limitada):
{dados_brutos}

Com base nas regras e no exemplo, gere a resposta final.<|im_end|>
<|im_start|>assistant
"""
)

# =========================
# Model Invocation Wrapper
# =========================
def invoke_llm(prompt_text):
    if not llm_client:
        raise ValueError("LLM client is not initialized.")
    try:
        response = llm_client.invoke(prompt_text)
        return response.content if hasattr(response, 'content') else response
    except Exception as e:
        print(f"❌ Error invoking LLM provider '{LLM_PROVIDER}': {e}")
        traceback.print_exc()
        return "Desculpe, ocorreu um erro ao contatar o modelo de linguagem."


# =========================
# Query Engine (with Self-Correction and TypeError Fix)
# =========================
def clean_generated_code(code_str: str) -> str:
    """Cleans markdown, backticks, and prefixes from the generated code string."""
    code_str = code_str.strip()
    if code_str.startswith("```python"):
        code_str = code_str[9:]
    if code_str.startswith("```"):
        code_str = code_str[3:]
    if code_str.endswith("```"):
        code_str = code_str[:-3]
    
    code_str = code_str.strip().replace('`', '')

    if not code_str.startswith("df_result"):
        code_str = "df_result = " + code_str
        
    return code_str

def query_structured_data(query: str, file_path: str):
    print(f"Executing intelligent structured data query on: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        for col in df.columns:
            if 'data' in col.lower() or 'date' in col.lower():
                try: 
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception: 
                    pass

        # --- ATTEMPT 1: Initial Code Generation ---
        print("🧠 Etapa 1/2: Gerando código inicial com LLM...")
        column_info = df.dtypes.to_string()
        code_gen_prompt = CODE_GENERATOR_PROMPT.format(
            columns_and_types=column_info, 
            question=query
        )
        generated_code = invoke_llm(code_gen_prompt)
        generated_code = clean_generated_code(generated_code)
        print(f"✅ Código Gerado (Tentativa 1):\n---\n{generated_code}\n---")
        
        result = None
        error_feedback = ""
        try:
            scope = {"df": df, "pd": pd}
            exec(generated_code, {}, scope)
            result = scope.get('df_result')

            if isinstance(result, (pd.DataFrame, pd.Series)) and result.empty:
                print("⚠️ Tentativa 1 retornou um resultado vazio. Tentando auto-correção.")
                error_feedback = "Nenhum resultado encontrado."
                result = None 
        except Exception as e:
            print(f"⚠️ Tentativa 1 falhou com erro. Tentando auto-correção. Erro: {e}")
            error_feedback = str(e)
            result = None

        # --- ATTEMPT 2: Self-Correction if Needed ---
        if result is None:
            print("\n🤖 Etapa 2/2: Iniciando auto-correção...")
            repair_prompt = CODE_REPAIR_PROMPT.format(
                question=query,
                failed_code=generated_code,
                error_feedback=error_feedback,
                df_head=df.head().to_string()
            )
            corrected_code = invoke_llm(repair_prompt)
            corrected_code = clean_generated_code(corrected_code)
            print(f"✅ Código Corrigido Gerado (Tentativa 2):\n---\n{corrected_code}\n---")
            
            scope = {"df": df, "pd": pd}
            exec(corrected_code, {}, scope)
            result = scope.get('df_result')
            generated_code = corrected_code 

        # ======================================================================
        # <<< FINAL RESULT PROCESSING BLOCK (TypeError FIX) >>>
        # ======================================================================
        
        # CASE 1: The result is a single number (from a .count(), .sum(), .shape[0] etc.)
        if isinstance(result, (int, float, np.number)):
            return {"total_count": int(result), "sample_df": pd.DataFrame()}
        
        # CASE 2: The result is a NumPy array (from a .unique() call)
        if isinstance(result, np.ndarray):
            df_sample_result = pd.DataFrame(result, columns=['Result'])
            return {
                "total_count": len(df_sample_result),
                "sample_df": df_sample_result,
                "output_columns": df_sample_result.columns.tolist(),
                "generated_code": generated_code
            }

        # CASE 3: The result is a DataFrame or a Series (standard query for data)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            df_sample_result = result
            if isinstance(df_sample_result, pd.Series):
                df_sample_result = df_sample_result.to_frame()
            if df_sample_result.empty:
                return "A consulta não retornou nenhum resultado, mesmo após a tentativa de correção."

            code_for_counting = re.sub(r'\.head\(\s*\d+\s*\)\s*$', '', generated_code)
            scope_count = {"df": df, "pd": pd}
            exec(code_for_counting, {}, scope_count)
            df_full_result = scope_count.get('df_result')
            total_count = len(df_full_result) if isinstance(df_full_result, (pd.DataFrame, pd.Series)) else 0
            
            return {
                "total_count": total_count, "sample_df": df_sample_result,
                "output_columns": df_sample_result.columns.tolist(), "generated_code": generated_code
            }
            
        return "A consulta não retornou um formato de resultado esperado."

    except Exception as e:
        print(f"❌ Erro irrecuperável em query_structured_data para '{file_path}': {e}")
        traceback.print_exc()
        return f"Ocorreu um erro ao processar o arquivo de dados: {e}"





# =========================
# Tool and System Initialization
# =========================




def query_philips_api(base_url: str, user_query: str = None, timeout: int = 8):
    """
    Fetches data from the Philips API endpoint (base_url).
    - base_url: full URL to call (e.g. Beeceptor endpoint).
    - user_query: optional string the user asked (kept for compatibility; currently unused).
    Returns a dict similar to query_structured_data:
      {
        "total_count": int,
        "sample_df": pd.DataFrame,
        "output_columns": [...],
        "raw_json": <original JSON>
      }
    """
    try:
        # Basic GET (you can extend to add headers, auth, params based on user_query)
        resp = requests.get(base_url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()  # could be dict, list, etc.

        # Normalize into DataFrame when appropriate
        if isinstance(payload, list):
            df = pd.json_normalize(payload)
        elif isinstance(payload, dict):
            # If the dict contains a main list, try common keys; otherwise normalize dict to single-row df
            if any(isinstance(v, list) for v in payload.values()):
                # pick the largest list value
                largest_list_key = max((k for k, v in payload.items() if isinstance(v, list)),
                                       key=lambda k: len(payload[k]), default=None)
                if largest_list_key:
                    df = pd.json_normalize(payload[largest_list_key])
                else:
                    df = pd.json_normalize(payload)
            else:
                df = pd.json_normalize(payload)
        else:
            # not JSON-serializable structure
            return {"error": "Resposta da API não está em JSON padrão."}

        # Convert date-like columns if present
        for col in df.columns:
            if 'date' in col.lower() or 'data' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass

        total_count = len(df)
        sample_df = df.head(20)

        return {
            "total_count": int(total_count),
            "sample_df": sample_df,
            "output_columns": sample_df.columns.tolist(),
            "generated_code": None,
            "raw_json": payload
        }

    except requests.exceptions.RequestException as re:
        print(f"❌ Philips API request failed: {re}")
        return f"Erro ao contatar a API Philips: {re}"
    except ValueError as ve:
        print(f"❌ Philips JSON parsing error: {ve}")
        return f"Erro ao interpretar resposta JSON da Philips: {ve}"
    except Exception as e:
        print(f"❌ Philips handler unexpected error: {e}")
        traceback.print_exc()
        return f"Erro interno ao processar dados da Philips: {e}"



def create_general_qa_tool(qa_chain):
    canned_answers = {
        # --- Respostas existentes ---
        "quem te desenvolveu": "Eu sou o Nostradecius, um assistente de IA factual desenvolvido por Vincent Pernarh, Vitoria Andrade e Decio Martins.",
        "qual o seu nome": "Meu nome é Nostradecius, seu assistente de IA.",
        "sobre voce": "Eu sou o Nostradecius, um assistente de IA factual. Minha função é responder perguntas usando a base de conhecimento interna. Fui desenvolvido por Vincent Pernarh, Vitoria Andrade e Decio Martins.",
        "sobre você": "Eu sou o Nostradecius, um assistente de IA factual. Minha função é responder perguntas usando a base de conhecimento interna. Fui desenvolvido por Vincent Pernarh, Vitoria Andrade e Decio Martins.",
        
        # --- CHAVES NOVAS E CORRIGIDAS ---
        "o que você faz": "Minha função é atuar como um assistente de IA para responder perguntas, consultar dados e fornecer informações com base nos documentos e bases de dados da empresa.",
        "o que voce faz": "Minha função é atuar como um assistente de IA para responder perguntas, consultar dados e fornecer informações com base nos documentos e bases de dados da empresa.",
        "o que você pode fazer": "Eu posso responder perguntas sobre a empresa, consultar dados de produtividade e listas de funcionários, e ajudar a encontrar informações em nossos documentos.",
        "o que voce pode fazer": "Eu posso responder perguntas sobre a empresa, consultar dados de produtividade e listas de funcionários, e ajudar a encontrar informações em nossos documentos.",
        "me fala tres coisas que voce pode fazer": "Claro! Eu posso: \n1. Consultar dados em planilhas (como projetos e funcionários). \n2. Responder perguntas sobre documentos internos. \n3. Fornecer informações gerais sobre a empresa.",

        # --- NOVAS RESPOSTAS SUGERIDAS ---
        "opa": "Olá! Como posso te ajudar?",
        "oi": "Olá! No que posso ajudar hoje?",
        "tudo bem?": "Tudo bem por aqui! Estou pronto para ajudar.",
        "tudo bem": "Tudo bem por aqui! Estou pronto para ajudar.",
        "obrigado": "De nada! Fico feliz em ajudar.",
        "obrigada": "De nada! Fico feliz em ajudar.",
        
        # --- CHAVE CORRIGIDA (com acento) ---
        "o que falamos até agora": "Eu não consigo resumir nossa conversa, mas você pode rolar para cima para ver nosso histórico."
    }
    def query_general_company_info(query: str) -> str:
        print("Executing general knowledge query.")
        # Padroniza a consulta para melhor correspondência de chave
        query_lower = query.lower().strip().replace('?', '').replace('!', '')
        
        # Procura por uma correspondência exata primeiro
        if query_lower in canned_answers:
            return canned_answers[query_lower]
        
        # Procura se a consulta contém a chave (como você já faz)
        for key, answer in canned_answers.items():
            if key in query_lower:
                return answer
        
        # Se não for enlatada, prossiga para a cadeia RAG
        try:
            return qa_chain.invoke(query)
        except Exception as e:
            print(f"Error during FAISS query: {e}")
            return "Ocorreu um erro ao consultar a base de conhecimento."
    return query_general_company_info

def initialize_system():
    global llm_client, embedding_model, tools, file_manifest
    try:
        print("--- 🚀 INICIANDO SISTEMA 🚀 ---")
        with open(FILE_MANIFEST_PATH, 'r', encoding='utf-8') as f:
            file_manifest = json.load(f)
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, cache_folder=EMBEDDING_CACHE_DIR)

        print(f"🔧 Configurando LLM Provider: {LLM_PROVIDER}")
        
        if LLM_PROVIDER == "GEMINI":
            if not GEMINI_API_KEY: raise ValueError("GEMINI_API_KEY not set.")
            llm_client = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY, 
                convert_system_message_to_human=True
            )
            print("✨ Cliente LangChain Gemini inicializado.")
        elif LLM_PROVIDER == "OPENAI":
            if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY not set.")
            llm_client = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
            print("✨ Cliente LangChain OpenAI inicializado.")
        elif LLM_PROVIDER == "LOCAL":
            if not os.path.exists(LLM_MODEL_PATH): raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
            llm_client = LlamaCpp(
                model_path=LLM_MODEL_PATH, temperature=0.0, n_ctx=4096,
                n_batch=512, n_threads=min(os.cpu_count() or 8, 11),
                verbose=False, stop=["<|im_end|>"]
            )
            print(f"✨ Modelo Local LlamaCpp carregado de: {LLM_MODEL_PATH}")
        
        else:
            raise ValueError(f"Provedor de LLM inválido: '{LLM_PROVIDER}'.")

        temp_tools = {}
        for item in file_manifest:
            tool_name, file_path = item["tool_name"], item["file_path"]
            
            if item.get("tool_type") == "vector_qa":
                print(f"   -> Criando ferramenta 'vector_qa' ({tool_name}) com a nova arquitetura LCEL.")
                vector_store = FAISS.load_local(file_path, embedding_model, allow_dangerous_deserialization=True)
                retriever = vector_store.as_retriever()

                retrieval_chain = (
                    {"context": retriever, "input": RunnablePassthrough()}
                    | GENERAL_QA_PROMPT
                    | llm_client
                    | StrOutputParser()
                )
                
                temp_tools[tool_name] = create_general_qa_tool(retrieval_chain)
            
            elif item.get("tool_type") == "dataframe_agent":
                print(f"   -> Criando ferramenta 'dataframe_agent' ({tool_name}).")
                temp_tools[tool_name] = lambda q, path=file_path: query_structured_data(q, path)
            elif item.get("tool_type") == "external_api":
                print(f"   -> Criando ferramenta 'external_api' ({tool_name}). Endpoint: {file_path}")
                # capture file_path in default arg to freeze its value in the lambda
                def make_api_tool(path):
                    return lambda q=None: query_philips_api(path, user_query=q)
                temp_tools[tool_name] = make_api_tool(file_path)
        
        tools = temp_tools
        print("--- ✨ SISTEMA PRONTO PARA USO ✨ ---")
    except Exception as e:
        print(f"❌ ERRO FATAL DURANTE A INICIALIZAÇÃO: {e}")
        traceback.print_exc()
        llm_client = embedding_model = None
        tools = {}

# =========================
# Flask Routes
# =========================
@app.route("/healthcheck")
def healthcheck():
    return jsonify({"status": "ready" if llm_client and tools else "initializing"}), 200 if llm_client and tools else 503

@app.route("/ask", methods=["POST"])
def ask_question():
    if not llm_client or not tools:
        return jsonify({"error": "O sistema ainda está inicializando. Por favor, aguarde."}), 503
    
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id")

    if not question: return jsonify({"error": "Nenhuma pergunta fornecida."}), 400
    if not session_id: return jsonify({"error": "Nenhum ID de sessão ('session_id') foi fornecido."}), 400

    print(f"\n\n--- Nova Pergunta Recebida (Sessão: {session_id}): '{question}' ---")
    
    if session_id not in conversation_memory_store:
        print(f"   -> Criando novo histórico de mensagens para a sessão {session_id}")
        conversation_memory_store[session_id] = ChatMessageHistory()

    chat_history_object = conversation_memory_store[session_id]
    chat_history = "\n".join([f"{'Human' if 'user' in msg.type else 'AI'}: {msg.content}" for msg in chat_history_object.messages])
    
    final_answer = ""
    try:
        with llm_lock:
            tools_description = "\n".join([f"- tool_name: {i['tool_name']}\n  description: {i['description']}" for i in file_manifest])
            router_prompt_formatted = ROUTER_PROMPT.format(
                tools_description=tools_description, question=question, chat_history=chat_history
            )
            llm_choice_raw = invoke_llm(router_prompt_formatted)
            chosen_tool_name = next((tool for tool in llm_choice_raw.split() if tool in tools), None)
            print(f"🤖 Roteador escolheu a ferramenta (com contexto): '{chosen_tool_name}'")
            
            tool_to_execute = tools.get(chosen_tool_name)
            if not tool_to_execute:
                final_answer = f"Desculpe, não consegui encontrar uma ferramenta adequada para responder a isso. Por favor, tente reformular sua pergunta."
            else:
                tool_result = tool_to_execute(question)
                print(f"📝 Saída Bruta da Ferramenta:\n---\n{tool_result}\n---")

                if isinstance(tool_result, str):
                    final_answer = tool_result
                elif isinstance(tool_result, dict):
                    sample_df = tool_result.get("sample_df", pd.DataFrame())
                    total_count = tool_result.get("total_count", 0)
                    dados_brutos_str = ""
                    if sample_df is not None and not sample_df.empty:
                        dados_brutos_str = sample_df.to_markdown(index=False)
                    final_prompt = SUMMARIZER_PROMPT.format(
                        dados_brutos=dados_brutos_str, question=question, total_count=total_count
                    )
                    final_answer = invoke_llm(final_prompt)
                else:
                    final_answer = "A ferramenta retornou um resultado em formato inesperado."
    
    except Exception as e:
        print(f"❌ Erro ao executar a ferramenta: {e}")
        traceback.print_exc()
        final_answer = "Ocorreu um erro durante a execução da sua solicitação."

    chat_history_object.add_user_message(question)
    chat_history_object.add_ai_message(final_answer)
    print(f"   -> Memória da sessão {session_id} foi atualizada.")

    print(f"💬 Resposta Final Programática:\n---\n{final_answer}\n---")

    def generate_response_stream():
        tokens = re.split(r'(\s+)', final_answer)
        for token in tokens:
            if token:
                yield f"data: {json.dumps(token)}\n\n"
                time.sleep(0.02)
        yield "data: [END]\n\n"

    return Response(generate_response_stream(), content_type="text/event-stream")

if __name__ == "__main__":
    print("Verifique se o seu 'file_manifest.json' tem 'tool_type' para cada ferramenta.")
    threading.Thread(target=initialize_system, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", port=5000)