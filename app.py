import os
import json
import time
import threading
import io
import re
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
# Ensure your model path is correct
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "C:/Users/perna/Desktop/NOSTRADECIUS/gemma-3-4b-it-UD-Q5_K_XL.gguf")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "models_cache")
FILE_MANIFEST_PATH = "file_manifest.json"

# =========================
# Globals for Loaded Models & Tools
# =========================
llm = None
embedding_model = None
# The 'tools' dictionary will map tool_name to a callable function
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
    """Você é o Nostradecius, um assistente de IA factual.O Nustradecius foi desenvolvido por Vincent Pernarh, Vitoria Andrade e Decio Martins
    Sua única função é responder à pergunta do usuário usando exclusivamente o CONTEXTO fornecido.

    --- Regras Estritas ---
    1. Responda em Português do Brasil.
    2. Se a informação não estiver no CONTEXTO, responda EXATAMENTE: 'Não encontrei dados relevantes sobre isso.'
    3. Baseie-se APENAS no CONTEXTO. Não use conhecimento externo.
    -----------------------

    CONTEXTO:
    {context}

    PERGUNTA DO USUÁRIO: {question}
    RESPOSTA:"""
)




# In your app.py, replace the COLUMN_SELECTOR_PROMPT

COLUMN_SELECTOR_PROMPT = PromptTemplate.from_template(
    """Sua tarefa é atuar como um seletor de colunas inteligente para um DataFrame. Dada a pergunta do usuário, você deve identificar TODAS as colunas relevantes para encontrar e para exibir uma resposta completa.

    Siga este processo de raciocínio:
    1.  **Intenção Principal:** Qual é o objetivo final do usuário? Ele está pedindo uma lista, um valor específico, uma confirmação?
    2.  **Colunas para Filtragem:** Identifique as entidades-chave na pergunta (ex: nomes como "Decio", status como "done"). Mapeie essas entidades para os nomes de colunas disponíveis (`{columns}`). Essas colunas são essenciais para filtrar os dados.
    3.  **Colunas de Informação:** Após a filtragem, qual informação o usuário realmente quer ver? A pergunta pode pedir dados explicitamente (ex: "liste os projetos") ou a necessidade pode ser implícita. Inclua colunas necessárias para construir uma resposta útil e completa, o que frequentemente inclui identificadores (como 'Nome do Projeto' ou 'ID do Funcionário') para dar contexto.
    4.  **Seleção Final:** Combine as colunas de filtragem e de informação em uma única lista. O objetivo é ser abrangente; é melhor incluir uma coluna um pouco menos relevante do que omitir uma importante.

    Ao final, retorne APENAS a lista Python com os nomes das colunas. Não inclua seu raciocínio ou qualquer outro texto.

    --- Exemplo ---
    Colunas Disponíveis: ['NProjeto', 'Nome', 'Status', 'Analista DHL', 'Custo']
    Pergunta do Usuário: "liste 2 projetos do Decio com status done"
    Seu Raciocínio (interno, não deve ser exibido):
    1.  **Intenção:** O usuário quer uma lista de projetos.
    2.  **Colunas para Filtragem:** A pergunta contém "Decio" (mapeia para 'Analista DHL') e "done" (mapeia para 'Status').
    3.  **Colunas de Informação:** O usuário quer ver os "projetos", então 'NProjeto' e 'Nome' são necessários para a identificação.
    4.  **Seleção Final:** ['NProjeto', 'Nome', 'Status', 'Analista DHL']
    Sua Resposta:
    ['NProjeto', 'Nome', 'Status', 'Analista DHL']
    
    --- Instruções Cruciais ---
    -   Baseie sua seleção *apenas* nas "Colunas Disponíveis" fornecidas abaixo para a pergunta atual.
    -   Sua resposta final deve ser exclusivamente uma lista Python de strings.

    Colunas Disponíveis: {columns}
    Pergunta do Usuário: "{question}"
    Sua Resposta:"""
)



# In your app.py, replace the old prompt with this one.

PANDAS_CODE_GENERATOR_PROMPT = PromptTemplate.from_template(
    """Sua tarefa é gerar um script Python curto usando a biblioteca pandas para responder a uma pergunta.
    Gere APENAS o código Python que filtra e imprime o resultado. NÃO inclua explicações ou comentários.

    --- INSTRUÇÃO CRÍTICA ---
    No seu código final, SELECIONE APENAS AS COLUNAS MAIS RELEVANTES para a pergunta do usuário. Não imprima todas as colunas.

    --- Informações do DataFrame ---
    Nome do DataFrame: `df`
    Colunas disponíveis para sua consulta: {columns}

    Primeiras 5 linhas das colunas relevantes:
    {head}
    ----------------------------------

    --- Exemplo ---
    Pergunta: "liste 3 projetos com status ongoing e seus nomes"
    Colunas disponíveis: ['NProjeto', 'Nome', 'Status', 'Analista', 'Custo','Analista DHL']
    Código Gerado:
    print(df[df['Status'].str.lower() == 'ongoing'][['NProjeto', 'Nome', 'Status','Analistas DHL']].head(3))
    -----------------

    PERGUNTA DO USUÁRIO: "{question}"

    CÓDIGO PYTHON GERADO:"""
)

# In your app.py, replace the FINAL_ANSWER_SYNTHESIZER_PROMPT

FINAL_ANSWER_SYNTHESIZER_PROMPT = PromptTemplate.from_template(
    """Sua tarefa é atuar como um assistente prestativo.
    Formule uma resposta clara e amigável em Português do Brasil com base nos dados brutos e na pergunta original do usuário.
    Se os dados brutos indicarem um erro ou que nada foi encontrado, diga isso de forma simples. Não invente informações.
    Respode baseado APENAS nos dados fornecidos.

    Dados Brutos:
    {tool_output}

    Pergunta Original:
    {question}

    Resposta Final:"""
)

# =========================
# Tool Definitions
# =========================

import ast # Add this import at the top of your file with the others

# Replace the whole function in your app.py

def query_structured_data(query: str, file_path: str) -> str:
    """
    Loads a structured file (CSV/XLSX) and uses a two-stage LLM process:
    1. Select relevant columns to reduce context size.
    2. Generate and execute Python code to answer the query using the reduced context.
    """
    print(f"Executing intelligent structured data query on: {file_path}")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # --- STAGE 1: Select Relevant Columns with Robust Parsing ---
        print("🧠 Etapa 1/2: Selecionando colunas relevantes...")
        col_selector_prompt = COLUMN_SELECTOR_PROMPT.format(
            columns=df.columns.tolist(),
            question=query
        )
        
        response_str = llm.invoke(col_selector_prompt).strip()
        relevant_columns = []

        # Attempt 1: The ideal case, parse a perfect Python list
        try:
            relevant_columns = ast.literal_eval(response_str)
        except (ValueError, SyntaxError):
            # Attempt 2: If the model added text, find quoted strings inside
            print("⚠️  ast.literal_eval falhou. Tentando extrair colunas com regex...")
            try:
                # This regex finds all substrings inside single or double quotes
                extracted_cols = re.findall(r"['\"]([^'\"]+)['\"]", response_str)
                if extracted_cols:
                    relevant_columns = extracted_cols
                else:
                    # Fallback if no columns could be extracted
                    print("⚠️ Regex não encontrou colunas. Usando todas as colunas como fallback final.")
                    relevant_columns = df.columns.tolist()
            except Exception:
                relevant_columns = df.columns.tolist()
        
        # Final Validation: De-duplicate and ensure extracted column names exist in the DataFrame
        unique_cols = list(dict.fromkeys(relevant_columns)) # De-duplicate while preserving order
        validated_columns = [col for col in unique_cols if col in df.columns]
        if not validated_columns:
            print("⚠️ Nenhuma coluna válida foi identificada após a validação. Usando todas as colunas.")
            relevant_columns = df.columns.tolist()
        else:
            relevant_columns = validated_columns
            print(f"✅ Colunas relevantes identificadas: {relevant_columns}")


        # --- STAGE 2: Generate Python code with Reduced Context ---
        print("🧠 Etapa 2/2: Gerando código com contexto reduzido...")
       
        # FIX: Prevent pandas from truncating the output data for the LLM
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

               
        reduced_head_markdown = df[relevant_columns].head().to_markdown()

        code_gen_prompt = PANDAS_CODE_GENERATOR_PROMPT.format(
            columns=relevant_columns,
            head=reduced_head_markdown,
            question=query
        )
        
        generated_code = llm.invoke(code_gen_prompt)
        cleaned_code = re.sub(r"```python\n|```", "", generated_code).strip()
        print(f"✅ Código Gerado:\n---\n{cleaned_code}\n---")

        # --- Execute the generated code ---
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(cleaned_code, {"df": df})
        
        result = buffer.getvalue()
        
        if not result.strip():
            return "A consulta não retornou nenhum resultado."
            
        return result

    except Exception as e:
        print(f"❌ Erro em query_structured_data para '{file_path}': {e}")
        return f"Ocorreu um erro ao processar o arquivo de dados: {e}"
    


    
# This function will be created dynamically during initialization
# because it depends on the QA chain which is loaded later.
def create_general_qa_tool(qa_chain) -> callable:
    """Factory function to create the general info query tool."""
    def query_general_company_info(query: str) -> str:
        """
        Queries the FAISS vector store for general, unstructured information.
        """
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
        
        # 1. Load File Manifest
        print("🔄 (1/4) Carregando manifesto de arquivos...")
        with open(FILE_MANIFEST_PATH, 'r', encoding='utf-8') as f:
            file_manifest = json.load(f)
        print("✅ (1/4) Manifesto carregado.")

        # 2. Load Embedding Model
        print("🔄 (2/4) Carregando modelo de embeddings...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, cache_folder=EMBEDDING_CACHE_DIR
        )
        print("✅ (2/4) Embeddings carregados.")

        # 3. Load LLM
        print("🔄 (3/4) Carregando LLM (LlamaCpp)...")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
        llm = LlamaCpp(
            model_path=LLM_MODEL_PATH,
            temperature=0.0,
            n_ctx=6096,
            n_batch=512,
            verbose=False,
            stop=["\nCÓDIGO", "\n---"] # Stop tokens to keep generation clean
        )
        print("✅ (3/4) LLM carregado.")

        # 4. Initialize Tools
        print("🔄 (4/4) Configurando ferramentas...")
        temp_tools = {}
        for item in file_manifest:
            tool_name = item["tool_name"]
            file_path = item["file_path"]

            if tool_name == "general_company_information":
                # This tool requires loading the FAISS index and creating a QA chain
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
                # Use the factory to create the tool function
                temp_tools[tool_name] = create_general_qa_tool(qa_chain)

            elif tool_name in ["productivity_data_query", "employee_list_query"]:
                # These are structured data tools. We create a lambda to call the generic
                # query_structured_data function with the specific file_path.
                 print(f"  -> Configurando ferramenta: {tool_name} para o arquivo {file_path}")
                 temp_tools[tool_name] = lambda q, path=file_path: query_structured_data(q, path)

        tools = temp_tools
        print("✅ (4/4) Todas as ferramentas estão prontas.")
        print("--- ✨ SISTEMA PRONTO PARA USO ✨ ---")

    except Exception as e:
        print(f"❌ ERRO FATAL DURANTE A INICIALIZAÇÃO: {e}")
        # Ensure globals are cleared on failure
        llm = embedding_model = None
        tools = {}


# =========================
# Flask Routes
# =========================

@app.route("/healthcheck")
def healthcheck():
    """Checks if the AI components are ready."""
    if llm and tools:
        return jsonify({"status": "ready", "available_tools": list(tools.keys())}), 200
    return jsonify({"status": "initializing"}), 503

@app.route("/ask", methods=["POST"])
def ask_question():
    """Handles user questions by routing to the correct tool and streaming the response."""
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