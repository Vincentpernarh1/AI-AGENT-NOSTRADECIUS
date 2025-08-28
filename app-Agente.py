import os
import json
import time
import threading
import io
import re
import ast
from contextlib import redirect_stdout
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# =========================
# Flask app & Configuration
# =========================
app = Flask(__name__)
CORS(app)

# --- Configuration ---
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "C:/Users/perna/Desktop/NOSTRADECIUS AGENT/gemma-3-4b-it-UD-Q5_K_XL.gguf")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "models_cache")
FILE_MANIFEST_PATH = "file_manifest.json"

# =========================
# Globals for Loaded Models & Tools
# =========================
llm = None
embedding_model = None
tools = {}
file_manifest = []

# =========================
# Prompts
# =========================

ROUTER_PROMPT = PromptTemplate.from_template(
    """Sua tarefa é agir como um roteador. Com base na pergunta do usuário, selecione a melhor ferramenta para respondê-la.
    Responda APENAS com o nome da ferramenta (`tool_name`). Não adicione nenhuma outra palavra ou explicação.

    --- Ferramentas Disponíveis ---
    {tools_description}
    --------------------------------

    Pergunta do Usuário: "{question}"

    Ferramenta Selecionada:"""
)

GENERAL_QA_PROMPT = PromptTemplate.from_template(
    """Você é o Nostradecius, um assistente de IA factual. O Nustradecius foi desenvolvido por Vincent Pernarh, Vitoria Andrade e Decio Martins.
    Sua única função é responder à pergunta do usuário usando exclusivamente o CONTEXTO fornecido.

    --- Regras Estritas ---
    1. Responda em Português do Brasil.
    2. Se a informação não estiver no CONTEXTO, responda EXATAMENTE: 'Não encontrei dados relevantes sobre isso.'
    3. Baseie-se APENAS no CONTEXTO. Não use conhecimento externo.
    
    CONTEXTO:
    {context}

    PERGUNTA DO USUÁRIO: {question}
    RESPOSTA:"""
)

# PROMPT 1 FOR STRUCTURED DATA: Selects relevant columns first.
COLUMN_SELECTOR_PROMPT = PromptTemplate.from_template(
    """Sua tarefa é identificar TODAS as colunas de um DataFrame estritamente necessárias para responder a uma pergunta.

    --- REGRAS CRÍTICAS ---
    1.  Você DEVE mapear TODAS as entidades da pergunta (nomes de pessoas, status, datas, etc.) para os nomes de colunas correspondentes.
    2.  Se a pergunta pede para "listar" itens (como "listar projetos"), você DEVE incluir a(s) coluna(s) que contêm o NOME ou a descrição principal desses itens (ex: 'Nome').

    --- Exemplo de Raciocínio Use para aprender o jeito de construir a saida não base só nesse opções. ---
    Pergunta: "liste 2 projetos do Decio com status done criados em 2025"
    
    Raciocínio: A pergunta pede para "listar projetos" (Regra 2), então preciso de 'Nome' e 'NProjeto'. Também menciona "Decio" (pessoa -> 'Analista DHL'), "done" (status -> 'Status'), e "2025" (data -> 'Data Criação').
    Resultado: ['NProjeto', 'Nome', 'Status', 'Analista DHL', 'Data Criação']

    Sua resposta final DEVE SER APENAS a lista Python de strings incluindo todas as colunas relevantes a query.

    --importante--
    Use as colunas disponivieis abaixo para responder não foque nos exemplos mas simm nas novas colunas.
    Colunas Disponíveis: {columns}
    Pergunta do Usuário: "{question}"
    Sua Resposta:"""
)

# PROMPT 2 FOR STRUCTURED DATA: Generates code based on the selected columns.
PANDAS_CODE_GENERATOR_PROMPT = PromptTemplate.from_template(
    """Sua tarefa é gerar uma ÚNICA linha de código Python para filtrar um DataFrame pandas e atribuir o resultado a uma variável `df_result`.

    --- REGRAS CRÍTICAS DE FILTRAGEM ---
    1.  O código gerado DEVE ser uma única linha e atribuir o resultado a `df_result`.
    2.  NÃO use `print()`.
    3.  Para nomes/texto (ex: 'Analista DHL'), use `.str.contains('valor', case=False, na=False)`.
    4. Para TODAS as colunas de texto (como 'Nome', 'Analista DHL' e 'Status','status'), use SEMPRE a busca flexível `.str.contains('valor', case=False, na=False)`. Isso garante que 'On Hold' e 'onhold' sejam encontrados.
    5.  Para DATAS (ex: 'Data Criação'), use o acessador `.dt.year` (ex: `df['Data Criação'].dt.year == 2025`).
    6.  Combine múltiplos filtros com `&` e parênteses.

    --- Informações do DataFrame ---
    Nome do DataFrame: `df`
    Colunas Relevantes Disponíveis: {columns}
    Primeiras 5 linhas das colunas relevantes:
    {head}
    ----------------------------------

    --- Exemplo Avançado ---
    Pergunta: "liste os 5 primeiros projetos do analista Decio com status On Going criados em 2025"
    Código Gerado:
    df_result = df[(df['Analista DHL'].str.contains('Decio', case=False, na=False)) & (df['Status'].str.contains('On Going', case=False, na=False)) & (df['Data Criação'].dt.year == 2025)]
    -----------------

    PERGUNTA DO USUÁRIO: "{question}"

    CÓDIGO PYTHON GERADO:"""
)

# Use this clean version in your app-Agente.py file
# from langchain.prompts import PromptTemplate


# Use this flexible and robust version in your Python code

FINAL_ANSWER_SYNTHESIZER_PROMPT = PromptTemplate.from_template(
    """Sua tarefa é atuar como um assistente de IA que formula uma resposta clara e amigável em Português do Brasil.
    Você deve usar os Dados Brutos para responder à Pergunta Original, seguindo as diretrizes e o exemplo fornecido.

    **Diretrizes de Formatação:**
    - NUNCA envolva sua resposta em blocos de código como ```markdown.
    - Use **texto em negrito** para destacar os pontos-chave.
    - Se a resposta for uma lista, use uma lista Markdown (`*`).
    - Ao listar projetos, inclua o número (`NProjeto`) e o nome (`Nome`) do projeto.
    - Deixe uma linha em branco antes e depois de qualquer lista.

    **Exemplo Perfeito com Dados Genéricos:**

    Dados Brutos:
    Total de resultados encontrados: 2
          NProjeto                           Nome                     Status
    0     YYYY_1234_ABC_DE   Nome do Projeto Exemplo 1 - Detalhes    On Hold
    1     YYYY_5678_XYZ_FG   Nome do Projeto Exemplo 2 - Outra área  On Hold

    Pergunta Original:
    Pode me listar os projetos do tipo ABC em hold?

   
    Claro! Encontrei 2 projetos do tipo **ABC** com status **"On Hold"**:

    * **YYYY_1234_ABC_DE:** Nome do Projeto Exemplo 1 - Detalhes
    * **YYYY_5678_XYZ_FG:** Nome do Projeto Exemplo 2 - Outra área

    **Sua Tarefa Agora:**

    Dados Brutos:
    {tool_output}

    Pergunta Original:
    {question}
    Resposta Final (use apenas Markdown puro): """
)


# =========================
# Helper Functions
# =========================
def is_structured_data_tool(tool_name):
    """Checks if the tool is for structured data."""
    return tool_name in ["productivity_data_query", "employee_list_query"]

# =========================
# Tool Definitions
# =========================

def query_structured_data(query: str, file_path: str) -> str:
    """
    Loads a structured file and uses a three-stage process to answer a query.
    """
    print(f"Executing intelligent structured data query on: {file_path}")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        else:
            df = pd.read_excel(file_path)
        
        for col in df.columns:
            if 'data' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    print(f"Aviso: Não foi possível converter a coluna '{col}' para data: {e}")

        # --- STAGE 1: Select Relevant Columns to prevent context overflow ---
        print("🧠 Etapa 1/3: Selecionando colunas relevantes...")
        col_selector_prompt = COLUMN_SELECTOR_PROMPT.format(
            columns=df.columns.tolist(), question=query
        )
        response_str = llm.invoke(col_selector_prompt).strip()
        relevant_columns = []
        try:
            relevant_columns = ast.literal_eval(response_str)
        except (ValueError, SyntaxError):
            print("⚠️  ast.literal_eval falhou. Tentando extrair colunas com regex...")
            extracted_cols = re.findall(r"['\"]([^'\"]+)['\"]", response_str)
            if extracted_cols:
                relevant_columns = extracted_cols
        
        unique_cols = list(dict.fromkeys(relevant_columns))
        validated_columns = [col for col in unique_cols if col in df.columns]
        
        if not validated_columns:
            print("⚠️ Nenhuma coluna válida foi identificada. Usando todas as colunas.")
            relevant_columns = df.columns.tolist()
        else:
            relevant_columns = validated_columns
            print(f"✅ Colunas relevantes identificadas: {relevant_columns}")

        # --- STAGE 2: Generate filtering code with a small, relevant context ---
        print("🧠 Etapa 2/3: Gerando código de filtragem...")
        pd.set_option('display.max_rows', 15)
        pd.set_option('display.max_columns', 30)
        pd.set_option('display.width', 400)

        reduced_head_markdown = df[relevant_columns].head().to_markdown()
        
        code_gen_prompt = PANDAS_CODE_GENERATOR_PROMPT.format(
            columns=relevant_columns, # Pass only the relevant columns as context
            head=reduced_head_markdown, 
            question=query
        )
        
        raw_generated_code = llm.invoke(code_gen_prompt)
        generated_code = raw_generated_code.strip().split('\n')[0]
        print(f"✅ Código Gerado:\n---\n{generated_code}\n---")

        # --- STAGE 3: Execute code and summarize results in Python ---
        print("🧠 Etapa 3/3: Executando e resumindo os resultados...")
        local_scope = {"df": df}
        exec(generated_code, {}, local_scope)
        df_result = local_scope.get('df_result')

        if df_result is None or df_result.empty:
            return "A consulta não retornou nenhum resultado."

        # After filtering, now we select the relevant columns for the final view
        df_final_view = df_result[relevant_columns]

        total_count = len(df_final_view)
        sample_df = df_final_view.head(10)

        summary_parts = [f"Total de resultados encontrados: {total_count}"]
        if total_count > 10:
            summary_parts.append("Mostrando uma amostra dos primeiros 10:")
        
        summary_parts.append(sample_df.to_string())
        
        return "\n".join(summary_parts)

    except Exception as e:
        print(f"❌ Erro em query_structured_data para '{file_path}': {e}")
        return f"Ocorreu um erro ao processar o arquivo de dados: {e}"

def create_general_qa_tool(qa_chain) -> callable:
    """Factory function to create the general info query tool."""
    def query_general_company_info(query: str) -> str:
        print("Executing general company information query.")
        try:
            result = qa_chain.invoke({"query": query})
            return result.get("result", "Não foi possível obter uma resposta.")
        except Exception as e:
            print(f"❌ Error in query_general_company_info: {e}")
            return "Ocorreu um erro ao consultar a base de conhecimento."
    return query_general_company_info


# =========================
# System Initialization
# =========================
def initialize_system():
    """Loads all models and configures the tools in a background thread."""
    global llm, embedding_model, tools, file_manifest
    try:
        print("--- 🚀 INICIANDO SISTEMA 🚀 ---")
        print("🔄 (1/4) Carregando manifesto de arquivos...")
        with open(FILE_MANIFEST_PATH, 'r', encoding='utf-8') as f:
            file_manifest = json.load(f)
        print("✅ (1/4) Manifesto carregado.")
        print("🔄 (2/4) Carregando modelo de embeddings...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, cache_folder=EMBEDDING_CACHE_DIR
        )
        print("✅ (2/4) Embeddings carregados.")
        print("🔄 (3/4) Carregando LLM (LlamaCpp)...")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
        llm = LlamaCpp(
            model_path=LLM_MODEL_PATH,
            temperature=0.0,
            n_ctx=4096,
            n_batch=512,
            verbose=False,
            stop=["\nCÓDIGO", "\n---", "Sua Resposta:"]
        )
        print("✅ (3/4) LLM carregado.")
        print("🔄 (4/4) Configurando ferramentas...")
        temp_tools = {}
        for item in file_manifest:
            tool_name = item["tool_name"]
            file_path = item["file_path"]
            if tool_name == "general_company_information":
                print(f"  -> Configurando ferramenta: {tool_name}")
                vector_store = FAISS.load_local(
                    file_path, embedding_model, allow_dangerous_deserialization=True
                )
                retriever = vector_store.as_retriever(search_kwargs={'k': 4})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff",
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": GENERAL_QA_PROMPT}
                )
                temp_tools[tool_name] = create_general_qa_tool(qa_chain)
            elif tool_name in ["productivity_data_query", "employee_list_query"]:
                 print(f"  -> Configurando ferramenta: {tool_name} para o arquivo {file_path}")
                 temp_tools[tool_name] = lambda q, path=file_path: query_structured_data(q, path)
        tools = temp_tools
        print("✅ (4/4) Todas as ferramentas estão prontas.")
        print("--- ✨ SISTEMA PRONTO PARA USO ✨ ---")
    except Exception as e:
        print(f"❌ ERRO FATAL DURANTE A INICIALIZAÇÃO: {e}")
        llm = embedding_model = None
        tools = {}

# =========================
# Flask Routes
# =========================
@app.route("/healthcheck")
def healthcheck():
    if llm and tools:
        return jsonify({"status": "ready", "available_tools": list(tools.keys())}), 200
    return jsonify({"status": "initializing"}), 503

@app.route("/ask", methods=["POST"])
def ask_question():
    if not llm or not tools:
        return jsonify({"error": "O sistema ainda está inicializando. Por favor, aguarde."}), 503
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Nenhuma pergunta fornecida."}), 400
    print(f"\n\n--- Nova Pergunta Recebida: '{question}' ---")
    tools_description = "\n".join(
        [f"- tool_name: {item['tool_name']}\n  description: {item['description']}" for item in file_manifest]
    )
    router_prompt_formatted = ROUTER_PROMPT.format(tools_description=tools_description, question=question)
    llm_choice_raw = llm.invoke(router_prompt_formatted)
    tool_name_match = re.search(r'\b\w+(?:_query|_information)\b', llm_choice_raw)
    if not tool_name_match:
        print(f"⚠️ Roteador indeciso. Resposta: '{llm_choice_raw}'. Usando fallback.")
        chosen_tool_name = "general_company_information"
    else:
        chosen_tool_name = tool_name_match.group(0).strip()
    print(f"🤖 Roteador escolheu a ferramenta: '{chosen_tool_name}'")
    tool_to_execute = tools.get(chosen_tool_name)
    if not tool_to_execute:
        print(f"❌ Ferramenta '{chosen_tool_name}' não encontrada.")
        return jsonify({"error": f"Erro interno: ferramenta '{chosen_tool_name}' não foi encontrada."}), 500
    try:
        raw_output = tool_to_execute(question)
        print(f"📝 Saída Bruta da Ferramenta:\n---\n{raw_output}\n---")
    except Exception as e:
        print(f"❌ Erro ao executar a ferramenta '{chosen_tool_name}': {e}")
        return jsonify({"error": "Ocorreu um erro durante a execução da ferramenta."}), 500
    
   
    final_prompt = FINAL_ANSWER_SYNTHESIZER_PROMPT.format(
        tool_output=raw_output, question=question
    )
    final_answer = llm.invoke(final_prompt)
    print(f"💬 Resposta Final Sintetizada:\n---\n{final_answer}\n---")
    def generate_response_stream():
        words = final_answer.split()
        for word in words:
            yield f"data: {word} \n\n"
            time.sleep(0.05)
        yield "data: [END]\n\n"
    return Response(generate_response_stream(), content_type="text/event-stream")

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    threading.Thread(target=initialize_system, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", port=5000)