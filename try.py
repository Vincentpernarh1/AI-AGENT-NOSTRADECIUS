import os
import json
import time
import threading
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import re 

# LangChain Imports
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# REMOVED: Unused experimental agent import
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# =========================
# Flask app, Config, Globals
# =========================
app = Flask(__name__)
CORS(app)

# --- Configuration ---
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "company_faiss_index")
# NOTE: You can continue using Mistral 7B as it's good enough for this JSON-based task.
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "C:/Users/perna/Desktop/NOSTRADECIUS AGENT/gemma-3-4b-it-UD-Q5_K_XL.gguf")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "models_cache")
DATA_FILES_PATH = "data_files"
FILE_MANIFEST_PATH = "file_manifest.json"

# --- Globals for Loaded Models ---
llm = None
vector_store = None
general_qa_chain = None
file_manifest = []

# =========================
# Prompts and Logic
# =========================

def get_router_prompt():
    """Creates the prompt to classify the user's intent."""
    return PromptTemplate.from_template(
        """Sua tarefa é classificar a pergunta do usuário em uma de duas categorias com base nos arquivos disponíveis.

        Arquivos Disponíveis:
        {file_descriptions}

        Categorias:
        1.  `data_file_query`: Se a pergunta do usuário parece estar diretamente relacionada a um dos arquivos descritos acima (ex: "qual a produtividade", "liste os projetos", "mostre os valores").
        2.  `general_query`: Se a pergunta for uma saudação, uma pergunta geral sobre a empresa, ou qualquer outra coisa que não se encaixe nos arquivos.

        Pergunta do Usuário: "{question}"

        Responda APENAS com a categoria (`data_file_query` or `general_query`).
        Categoria:"""
    )

def get_general_qa_prompt():
    """Prompt for the RAG chain that answers general questions."""
    template = (
        "Você é o Nostradecius, um assistente de IA.\n"
        "Responda SOMENTE em Português, de forma direta e concisa.\n\n"
        "REGRAS:\n"
        "1) Responda apenas com base no CONTEXTO fornecido.\n"
        "2) Se a resposta não estiver no contexto, diga exatamente: 'Não encontrei dados relevantes sobre isso.'\n"
        "3) Se a pergunta for sobre a SUA identidade (ex.: 'quem é você'), "
        "responda : 'Eu sou o Nostradecius, seu assistente de IA. Fui Desenvolvido por: Vincent Pernarh, Vitoria Andrade & Decio Martins'\n\n"
        "CONTEXTO:\n{context}\n\n"
        "Pergunta do usuário: {question}\n\n"
        "Resposta final (apenas a resposta direta):"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])


def get_data_summarization_prompt():
    """Prompt to format the final data result into natural language."""
    template = """
    Sua tarefa é responder à pergunta do usuário de forma amigável e em linguagem natural, utilizando os dados fornecidos no contexto.
    Não se limite a repetir os dados brutos. Formule uma frase completa e clara. Se for uma lista, formate-a de maneira legível.

    REGRAS:
    - Responda SOMENTE em Português.
    - Seja direto e prestativo.
    - Se o contexto estiver vazio ou não contiver informações relevantes, diga "Não encontrei resultados para a sua pesquisa.".

    CONTEXTO (Dados encontrados no arquivo):
    {data_context}

    PERGUNTA ORIGINAL DO USUÁRIO:
    {question}

    RESPOSTA FINAL (em linguagem natural e amigável):
    """
    return PromptTemplate(template=template, input_variables=["data_context", "question"])

# ===============================================
# NEW: Structured Data Query Function (Replaces the Agent)
# ===============================================
# ===============================================
# ATUALIZADO: Structured Data Query Function
# ===============================================

# ===============================================
# ATUALIZADO: Structured Data Query Function
# ===============================================

def query_data_file_structured(file_path: str, question: str) -> str:
    """
    Analyzes a data file in a two-step process:
    1. LLM generates a JSON plan to filter data.
    2. Python executes the plan and gets the data.
    3. LLM summarizes the resulting data into a natural language answer.
    """
    print(f"--- Analyzing file with structured query: {file_path} ---")
    if not os.path.exists(file_path):
        return f"Erro: O arquivo '{file_path}' não foi encontrado."

    try:
        df = pd.read_excel(file_path) if file_path.endswith(('.xlsx', '.xls')) else pd.read_csv(file_path)
        
        # --- PASSO 1: Gerar o plano JSON (lógica existente) ---
        json_prompt_template = """
        Your task is to create a JSON object to query a pandas DataFrame based on the user's question.
        Analyze the user's question and the available columns to create the filters.

        GUIDELINES:
        - The JSON output MUST be a single JSON object. Do not add any text before or after it.
        - The JSON object must contain ONLY the following keys: "filters", "limit", and "return_columns".
        - The "filters" key must contain a list of objects, where each object has "column", "operator", and "value".
        - The "column" in a filter must EXACTLY match one of the available column names.
        - The "operator" can only be '==', '!=', or 'contains'.
        - If the user asks for a specific number of items (e.g., "list 3 projects"), add a "limit" key.
        - If the user asks for specific columns to be returned, add a "return_columns" key with a list of column names.

        Available Columns: {df_columns}
        
        Example:
        User Question: "listar 2 projetos com status ongoing que incluir Decio Martins"
        JSON Output:
        {{
          "filters": [
            {{ "column": "Status", "operator": "==", "value": "ongoing" }},
            {{ "column": "Analista DHL", "operator": "contains", "value": "Decio Martins" }}
          ],
          "limit": 2,
          "return_columns": ["NProjeto", "Nome", "Status", "Analista DHL"]
        }}

        User Question: "{question}"

        JSON Output:
        """
        prompt = PromptTemplate.from_template(json_prompt_template)
        formatted_prompt = prompt.format(df_columns=df.columns.tolist(), question=question)

        print("🤔 Passo 1/3: Gerando plano de consulta JSON...")
        response_str = llm.invoke(formatted_prompt)
        
        # --- NOVO BLOCO TRY-EXCEPT PARA TRATAMENTO DE ERRO ---
        try:
            json_str_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if not json_str_match:
                print("❌ Erro: O LLM não retornou um JSON válido.")
                return "Não consegui entender como buscar os dados para sua pergunta."

            json_str = json_str_match.group(0)
            query_plan = json.loads(json_str)
            print(f"✅ Plano JSON gerado: {query_plan}")
        except json.JSONDecodeError as e:
            # Em caso de erro de JSON, imprima a resposta bruta do LLM para depuração
            print(f"❌ Ocorreu um erro ao decodificar o JSON: {e}")
            print(f"❌ Resposta crua do LLM que causou o erro:\n{response_str}")
            # Retorne uma mensagem amigável para o usuário
            return "Ocorreu um erro interno ao processar sua solicitação. Por favor, tente de novo com uma pergunta diferente."
        # --- FIM DO NOVO BLOCO ---

        # O restante do código permanece o mesmo
        # --- PASSO 2: Executar o plano em Python ---
        print("🚀 Passo 2/3: Executando o plano JSON...")
        
        result_df = df
        
        for f in query_plan.get("filters", []):
            col, op, val = f['column'], f['operator'], f['value']
            if col in result_df.columns:
                if op == '==':
                    result_df = result_df[result_df[col].astype(str).str.lower() == str(val).lower()]
                elif op == '!=':
                    result_df = result_df[result_df[col].astype(str).str.lower() != str(val).lower()]
                elif op == 'contains':
                    result_df = result_df[result_df[col].astype(str).str.contains(str(val), case=False, na=False)]

        return_cols = query_plan.get("return_columns", [])
        if return_cols:
            valid_return_cols = [c for c in return_cols if c in result_df.columns]
            if valid_return_cols:
                result_df = result_df[valid_return_cols]
        
        limit = query_plan.get("limit")
        if limit and isinstance(limit, int):
            result_df = result_df.head(limit)

        if result_df.empty:
            return "Não foram encontrados resultados para a sua pesquisa."
        
        # --- PASSO 3: Usar o LLM para sumarizar os resultados em linguagem natural ---
        print("✍️ Passo 3/3: Formulando a resposta final...")
        
        # Converte o DataFrame resultante em uma string para usar como contexto
        data_context_string = result_df.to_string(index=False)
        
        # Pega o prompt de sumarização
        summarization_prompt = get_data_summarization_prompt().format(
            data_context=data_context_string,
            question=question
        )
        
        # Invoca o LLM para a resposta final
        final_answer = llm.invoke(summarization_prompt)
        
        return final_answer.strip()

    except Exception as e:
        # Este 'catch' mais genérico ainda é útil para outros erros
        print(f"❌ Ocorreu um erro ao processar sua pergunta: {e}")
        return f"Ocorreu um erro ao processar sua pergunta no arquivo: {e}" 
# Model and Pipeline Initialization
# =========================
def initialize_pipelines():
    """Loads all necessary components in a background thread."""
    global llm, vector_store, general_qa_chain, file_manifest
    try:
        # 1. Load File Manifest
        print("🔄 1/5: Carregando manifesto de arquivos...")
        if os.path.exists(FILE_MANIFEST_PATH):
            with open(FILE_MANIFEST_PATH, 'r', encoding='utf-8') as f:
                file_manifest = json.load(f)
        print("✅ 1/5: Manifesto de arquivos carregado.")

        # 2. Load Embedding Model
        print("🔄 2/5: Carregando modelo de embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, cache_folder=EMBEDDING_CACHE_DIR)
        print("✅ 2/5: Modelo de embeddings carregado.")

        # 3. Load FAISS Vector Store
        print("🔄 3/5: Carregando índice vetorial FAISS...")
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ 3/5: Índice FAISS carregado.")

        # 4. Load LLM
        print("🔄 4/5: Carregando o LLM (LlamaCpp)...")
        llm = LlamaCpp(model_path=LLM_MODEL_PATH, temperature=0.2, n_ctx=8192, n_batch=512, verbose=False)
        print("✅ 4/5: LLM carregado.")

        # 5. Create General QA Chain
        print("🔄 5/5: Criando a pipeline de QA Geral...")
        prompt = get_general_qa_prompt()
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        general_qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt}
        )
        print("✅ 5/5: Todos os sistemas estão prontos!")

    except Exception as e:
        print(f"❌ Erro fatal durante a inicialização: {e}")
        llm = vector_store = general_qa_chain = None
        file_manifest = []

# =========================
# Flask Routes
# =========================
@app.route("/healthcheck")
def healthcheck():
    """Confirms if the pipelines are loaded and ready."""
    if general_qa_chain and llm:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "initializing"}), 503

@app.route("/ask", methods=["POST"])
def ask_question():
    """Main endpoint for the lightweight agent."""
    if not general_qa_chain or not llm:
        return jsonify({"error": "O sistema ainda está inicializando."}), 503

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Nenhuma pergunta fornecida."}), 400

    # --- Lightweight Agent Logic ---
    file_descriptions = "\n".join([f"- {item['file_path']}: {item['description']}" for item in file_manifest])
    router_prompt = get_router_prompt().format(file_descriptions=file_descriptions, question=question)
    
    print(f"🤔 Classificando a pergunta: '{question}'")
    intent = llm.invoke(router_prompt).strip().lower()
    print(f"✅ Intenção classificada como: '{intent}'")

    def generate_response():
        """Generates and streams the response based on the classified intent."""
        answer = ""
        try:
            if "data_file_query" in intent and file_manifest:
                # This logic could be improved to find the best file, but for now it's simple
                best_match = file_manifest[0] 
                # CALL THE NEW, RELIABLE FUNCTION
                answer = query_data_file_structured(best_match['file_path'], question)
            else: # Default to general_query
                response = general_qa_chain.invoke({"query": question})
                answer = response.get('result', "Não foi possível processar sua pergunta.")

            print(f"✅ Resposta gerada:\n{answer}")
            # Stream the answer back
                   
            words = answer.split()
            
            # print("words", words)
            for i in words:
                yield f"data: {i}\n\n"
                time.sleep(0.05)
            yield f"data: [END]\n\n"

        except Exception as e:
            print(f"❌ Erro durante a geração da resposta: {e}")
            error_message = f"Ocorreu um erro: {str(e)}"
            yield f"data: {json.dumps({'token': error_message})}\n\n"
            yield f"data: [END]\n\n"

    return Response(generate_response(), content_type="text/event-stream")

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    initialization_thread = threading.Thread(target=initialize_pipelines, daemon=True)
    initialization_thread.start()
    app.run(debug=True, host="0.0.0.0", port=5000)