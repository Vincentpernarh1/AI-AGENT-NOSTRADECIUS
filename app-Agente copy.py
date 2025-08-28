import os
import json
import time
import threading
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# LangChain Imports
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Other Libraries
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from unstructured.partition.auto import partition

# =========================
# Flask app, Config, Globals
# =========================
app = Flask(__name__)
CORS(app)

# --- Configuration ---
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "company_faiss_index")
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    "C:/Users/perna/Desktop/NOSTRADECIUS AGENT/qwen2-7b-instruct-q5_k_M.gguf"
)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "models_cache")
DATA_FILES_PATH = "data_files"
FILE_MANIFEST_PATH = "file_manifest.json"

# --- Globals for Loaded Models ---
llm = None
vector_store = None
general_qa_chain = None
file_manifest = []
embeddings = None
dataframe_cache = {}   # cache for loaded DataFrames

# --- File Embedding Cache ---
file_embedding_cache = {}  # { file_path: { "embedding": vec, "timestamp": epoch_time } }
EMBEDDING_TTL = 3600  # 1 hour

# =========================
# Prompts and Logic
# =========================
def get_router_prompt():
    return PromptTemplate.from_template(
        """Sua tarefa é classificar a pergunta do usuário em uma de duas categorias com base nos arquivos disponíveis.

        Arquivos Disponíveis:
        {file_descriptions}

        Categorias:
        1.  `data_file_query`: Se a pergunta do usuário parece estar diretamente relacionada a um dos arquivos descritos acima ("qual a produtividade ou valor ou projetos").
        2.  `general_query`: Se a pergunta for uma saudação, uma pergunta geral sobre a empresa, ou qualquer outra coisa que não se encaixe nos arquivos.

        Pergunta do Usuário: "{question}"

        Responda APENAS com a categoria (`data_file_query` or `general_query`).
        Categoria:"""
    )

def get_general_qa_prompt():
    template = (
        "Você é o Nostradecius, um assistente de IA.\n"
        "Responda SOMENTE em Português, de forma direta e concisa.\n\n"
        "REGRAS:\n"
        "1) Responda apenas com base no CONTEXTO fornecido.\n"
        "2) Se a resposta não estiver no contexto, diga exatamente: 'Não encontrei dados relevantes sobre isso.'\n"
        "3) Não adicione informações externas.\n"
        "4) Não repita a pergunta.\n"
        "5) Não use prefixos como 'PERGUNTA:' ou 'RESPOSTA:'. Apenas dê a resposta final.\n\n"
        "6) Se a pergunta for sobre a SUA identidade (ex.: 'quem é você', 'qual seu nome', 'o que você é'), "
        "responda : 'Eu sou o Nostradecius, seu assistente de IA. eu fui Desenvolvido por: Vincent Pernarh, Vitoria Andrade & Decio Martins'\n"
        "CONTEXTO:\n{context}\n\n"
        "Pergunta do usuário: {question}\n\n"
        "Resposta final (apenas a resposta direta):"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])

# =========================
# File Selection & Loading
# =========================
def embed_text(text: str):
    """Get embedding vector for a given text using global embeddings model."""
    return embeddings.embed_query(text)

def get_file_embedding(file: dict):
    """Return embedding for a file description with 1h cache expiry."""
    file_path = file['file_path']
    now = time.time()

    # Check cache
    if file_path in file_embedding_cache:
        entry = file_embedding_cache[file_path]
        if now - entry['timestamp'] < EMBEDDING_TTL:
            return entry['embedding']

    # Compute fresh embedding
    desc_vec = embed_text(file['description'])
    file_embedding_cache[file_path] = {"embedding": desc_vec, "timestamp": now}
    return desc_vec

def find_best_file(question: str, file_manifest: list):
    """Finds the most relevant file for a given question using semantic similarity."""
    if not file_manifest:
        return None

    q_vec = embed_text(question)
    best_match, best_score = None, -1

    for file in file_manifest:
        desc_vec = get_file_embedding(file)
        score = cosine_similarity([q_vec], [desc_vec])[0][0]
        if score > best_score:
            best_match, best_score = file, score

    if best_match:
        print(f"📂 Melhor arquivo encontrado: {best_match['file_path']} (similaridade={best_score:.3f})")
    return best_match

def load_dataframe(file_path: str):
    """Loads a DataFrame with caching to speed up repeated access."""
    if file_path in dataframe_cache:
        return dataframe_cache[file_path]

    if file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = None

    if df is not None:
        dataframe_cache[file_path] = df
    return df

# =========================
# Tool Functions
# =========================
def query_data_file(file_path: str, question: str) -> str:
    print(f"--- Analyzing file: {file_path} ---")
    if not os.path.exists(file_path):
        return f"Erro: O arquivo '{file_path}' não foi encontrado."

    filename = os.path.basename(file_path)
    try:
        df = load_dataframe(file_path)
        if df is None:
            return f"Formato de arquivo '{filename}' não suportado para análise."

        # STEP 1: Planner
        column_list = df.columns.tolist()
        planner_prompt = f"""
        Given the user's question, which of the following columns are needed to answer it?
        User Question: "{question}"
        Column List: {column_list}

        Return your answer as a Python-compatible list of strings. For example: ['column1', 'column2']
        Relevant Columns:
        """
        print("🤔 Step 1/2: Planning columns...")
        response = llm.invoke(planner_prompt)

        try:
            import ast
            relevant_columns = ast.literal_eval(response.strip())
            print(f"✅ Planner identified relevant columns: {relevant_columns}")
            filtered_df = df[relevant_columns]
        except Exception as e:
            print(f"⚠️ Could not parse planner response, using full dataframe. Error: {e}")
            filtered_df = df

        # STEP 2: Executor
        print("🚀 Step 2/2: Executing query...")
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            filtered_df,
            verbose=False,
            allow_dangerous_code=True
        )
        return pandas_agent.invoke({"input": question}).get(
            'output',
            'Não foi possível obter uma resposta do arquivo.'
        )

    except Exception as e:
        return f"Ocorreu um erro ao analisar o arquivo {filename}: {e}"

# =========================
# Initialization
# =========================
def initialize_pipelines():
    global llm, vector_store, general_qa_chain, file_manifest, embeddings
    try:
        # 1. File Manifest
        print("🔄 1/5: Carregando manifesto de arquivos...")
        if os.path.exists(FILE_MANIFEST_PATH):
            with open(FILE_MANIFEST_PATH, 'r', encoding='utf-8') as f:
                file_manifest = json.load(f)
        print("✅ 1/5: Manifesto carregado.")

        # 2. Embedding Model
        print("🔄 2/5: Carregando modelo de embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder=EMBEDDING_CACHE_DIR
        )
        print("✅ 2/5: Modelo de embeddings carregado.")

        # 3. FAISS Vector Store
        print("🔄 3/5: Carregando índice vetorial FAISS...")
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ 3/5: Índice FAISS carregado.")

        # 4. LLM
        print("🔄 4/5: Carregando o LLM...")
        llm = LlamaCpp(
            model_path=LLM_MODEL_PATH,
            temperature=0.2,
            n_ctx=8192,
            n_batch=512,
            verbose=False
        )
        print("✅ 4/5: LLM carregado.")

        # 5. General QA Chain
        print("🔄 5/5: Criando QA chain...")
        prompt = get_general_qa_prompt()
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        general_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        print("✅ 5/5: Pipelines prontas!")

    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        llm = vector_store = general_qa_chain = embeddings = None
        file_manifest = []

# =========================
# Flask Routes
# =========================
@app.route("/healthcheck")
def healthcheck():
    if general_qa_chain and llm:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "initializing"}), 503

@app.route("/ask", methods=["POST"])
def ask_question():
    if not general_qa_chain or not llm:
        return jsonify({"error": "O sistema ainda está inicializando."}), 503

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Nenhuma pergunta fornecida."}), 400

    # Intent classification
    file_descriptions = "\n".join([f"- {item['file_path']}: {item['description']}" for item in file_manifest])
    router_prompt = get_router_prompt().format(file_descriptions=file_descriptions, question=question)

    print(f"🤔 Classificando pergunta: '{question}'")
    intent = llm.invoke(router_prompt).strip().lower()
    print(f"✅ Intenção: '{intent}'")

    def generate_response():
        answer = ""
        try:
            if "data_file_query" in intent and file_manifest:
                best_match = find_best_file(question, file_manifest)
                if best_match:
                    answer = query_data_file(best_match['file_path'], question)
                else:
                    answer = "Não encontrei um arquivo relevante para sua pergunta."
            else:
                response = general_qa_chain.invoke({"query": question})
                answer = response.get('result', "Não foi possível processar sua pergunta.")

            print(f"✅ Resposta: {answer}")
            words = answer.split()
            for i in words:
                yield f"data: {i}\n\n"
                time.sleep(0.05)
            yield f"data: [END]\n\n"

        except Exception as e:
            print(f"❌ Erro durante a geração: {e}")
            yield f"data: {json.dumps({'token': str(e)})}\n\n"
            yield f"data: {json.dumps({'token': '[END]'})}\n\n"

    return Response(generate_response(), content_type="text/event-stream")

# =========================
# Main
# =========================
if __name__ == "__main__":
    initialization_thread = threading.Thread(target=initialize_pipelines, daemon=True)
    initialization_thread.start()
    app.run(debug=True, host="0.0.0.0", port=5000)
