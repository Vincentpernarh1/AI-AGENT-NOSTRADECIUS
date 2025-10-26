import os
import json
import time
import threading
import re
import pandas as pd
import difflib
import traceback
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
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


# Pandas display options
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None)


# --- Configuration ---
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "C:/Users/perna/Desktop/NOSTRADECIUS/gemma-3-4b-it-UD-Q5_K_XL.gguf")
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
# Prompts (Maintained as requested)
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

COLUMN_SELECTOR_PROMPT = PromptTemplate.from_template(
    """
Dada a pergunta do usuário: "{question}", selecione todas as colunas da lista abaixo que possam ser necessárias ou relevantes para responder à pergunta, considerando relações diretas e indiretas.

INSTRUÇÕES ADICIONAIS:
- Para cada valor, nome, status, data ou entidade mencionada na pergunta, inclua qualquer coluna cujo nome possivelmente armazene essa informação, mesmo que o nome da coluna não coincida exatamente com o termo da pergunta (por exemplo: se a pergunta mencionar um analista, inclua colunas como 'Nome Analista', 'Analista', etc).
- Inclua sempre as colunas necessárias para filtrar, identificar, ou responder partes específicas da pergunta, mesmo que a relação seja indireta.
- Se houver dúvida, seja inclusivo: prefira selecionar mais colunas do que menos.

Colunas disponíveis: {columns}

Responda apenas com uma lista Python contendo todos os nomes das colunas selecionadas. Não adicione explicações, apenas a lista.
"""
)

PROMPT_PREENCHER_LACUNAS = PromptTemplate.from_template(
    """Sua tarefa é completar um objeto JSON extraindo valores da pergunta do usuário.
Para cada chave no "JSON com Lacunas" fornecido, encontre o valor correspondente na "Pergunta do Usuário" e preencha a string vazia.
Se não for possível encontrar um valor para uma chave na pergunta, deixe seu valor como uma string vazia "".
Responda APENAS com o objeto JSON completo.

--- Exemplo ---
Pergunta do Usuário: "mostre-me itens para Jane Doe que estão ativos desde o ano de 2023"
JSON com Lacunas:
{{
  "alguma_coluna_de_nome": "",
  "alguma_coluna_de_status": "",
  "alguma_coluna_de_data": ""
}}

JSON Completo:
{{
  "alguma_coluna_de_nome": "Jane Doe",
  "alguma_coluna_de_status": "ativo",
  "alguma_coluna_de_data": "2023"
}}
---

**Tarefa Atual:**
Pergunta do Usuário: "{question}"

JSON com Lacunas:
{json_blanks}

JSON Completo:
"""
)



SUMMARIZER_PROMPT = PromptTemplate.from_template(
    """Você é um assistente que resume dados de forma clara e útil. Com base nos DADOS e na PERGUNTA do usuário, gere uma resposta natural em Português.

INSTRUÇÕES:
1.  Use APENAS as informações dos "DADOS ENCONTRADOS". Não invente nada.
2.  Escolha o MELHOR formato Markdown para os dados (parágrafo, lista ou tabela).
3.  Destaque palavras-chave importantes em negrito (`**palavra**`).

--- DADOS ENCONTRADOS ---
{dados_brutos}
--------------------------

PERGUNTA DO USUÁRIO: {question}
RESPOSTA:"""
)
# =========================
# Utility Functions
# =========================

def extract_limit_from_question(question):
    match = re.search(r'(listar|mostre|mostra|apenas|somente|exibir|exiba|mostrar|mínimo|minimo)\s+(\d+)', question, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return 20

# Maintained as requested
def get_best_match(user_value, unique_values):
    user_value_clean = str(user_value).strip().lower()
    clean_values = [str(v).strip().lower() for v in unique_values if pd.notnull(v)]
    matches = difflib.get_close_matches(user_value_clean, clean_values, n=1, cutoff=0.6)
    if matches:
        idx = clean_values.index(matches[0])
        return str(unique_values[idx])
    return user_value

# <<< REMOVED: `select_columns_programmatically` is no longer needed. >>>

# <<< REFACTORED: This is the new, more flexible query engine. >>>
def query_structured_data(query: str, file_path: str):
    print(f"Executing intelligent structured data query on: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        for col in df.columns:
            if 'data' in col.lower() or 'date' in col.lower():
                try: df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception: pass

        # --- STEP 1: LLM-Powered Column Selection ---
        print("🧠 Etapa 1/3: Selecionando colunas relevantes com LLM...")
        col_selector_prompt = COLUMN_SELECTOR_PROMPT.format(columns=df.columns.tolist(), question=query)
        response_str = llm.invoke(col_selector_prompt)
        print(f"  -> Resposta do LLM para seleção de colunas: {response_str}")
        
        relevant_columns = []
        try:
            # The LLM should return a string representation of a list, so we can evaluate it
            relevant_columns = eval(response_str.strip())

            if not isinstance(relevant_columns, list):
                raise TypeError("LLM response was not a list")
        except Exception as e:
            # print(f"⚠️ Aviso: Não foi possível decodificar a lista de colunas do LLM ({e}). Tentando extrair com regex.")
            relevant_columns = re.findall(r"['\"](.*?)['\"]", response_str)

        if not relevant_columns:
            return "Falha na Etapa 1: O modelo não conseguiu identificar colunas relevantes."
        
        # Validate that the selected columns actually exist to prevent errors
        validated_columns = list(dict.fromkeys([col for col in relevant_columns if col in df.columns]))
        print(f"✅ Colunas relevantes identificadas: {validated_columns}")
        if not validated_columns:
            return "Falha na Etapa 1: As colunas identificadas não existem no arquivo."
        relevant_columns = validated_columns


        # --- STEP 2: LLM-Powered Filter Extraction ---
        print("🧠 Etapa 2/3: Extraindo filtros da pergunta...")
        # Dynamically determine which columns are good candidates for filtering
        filterable_columns = [col for col in relevant_columns if df[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df[col])]
        json_blanks = {col: "" for col in filterable_columns}
        
        extracted_filters = {}
        if json_blanks:
            fill_prompt = PROMPT_PREENCHER_LACUNAS.format(question=query, json_blanks=json.dumps(json_blanks, indent=2))
            llm_response_str = llm.invoke(fill_prompt)
            try:
                match = re.search(r'(\{.*?\})', llm_response_str, re.DOTALL)
                if match:
                    raw_filters = json.loads(match.group(0))
                    extracted_filters = {k: v for k, v in raw_filters.items() if v and k in df.columns}
                    print(f"✅ Filtros extraídos: {extracted_filters}")
            except Exception as e:
                print(f"⚠️ Erro ao decodificar JSON de filtros: {e}")

        # --- STEP 3: Dynamic Query Building & Execution ---
        print("🧠 Etapa 3/3: Construindo e executando a consulta...")
        filter_conditions = []
        
        if extracted_filters:
            # --- Build a precise query if filters were found ---
            print("  -> Modo: Consulta baseada em filtros.")
            for col, value in extracted_filters.items():
                if pd.api.types.is_datetime64_any_dtype(df[col]) and re.fullmatch(r'\d{4}', str(value)):
                    filter_conditions.append(f"(df['{col}'].dt.year == {value})")
                elif df[col].dtype == 'object':
                    best_value = get_best_match(str(value), df[col].dropna().unique())
                    if str(value).lower() != str(best_value).lower(): print(f"  -> Corrigindo '{value}' para '{best_value}'")
                    val_escaped = str(best_value).replace("'", "\\'")
                    filter_conditions.append(f"(df['{col}'].str.contains('{val_escaped}', case=False, na=False,regex=False))")
        else:
            # --- Fallback to general text search if no filters were found ---
            print("  -> Modo: Fallback para busca geral de texto.")
            search_terms = query.split() # Simple split for general search
            general_conditions = []
            text_columns = [col for col in relevant_columns if df[col].dtype == 'object']
            for term in search_terms:
                for col in text_columns:
                    term_escaped = term.replace("'", "\\'")
                    general_conditions.append(f"(df['{col}'].str.contains('{term_escaped}', case=False, na=False))")
            if general_conditions:
                filter_conditions.append(f"({' | '.join(general_conditions)})")

        # --- Execution ---
        if filter_conditions:
            generated_code = f"df_result = df[{' & '.join(filter_conditions)}]"
        else:
            generated_code = "df_result = df" # No filters, return all data

        print(f"✅ Código Gerado:\n---\n{generated_code}\n---")
        
        local_scope = {"df": df}
        exec(generated_code, {}, local_scope)
        df_result = local_scope.get('df_result')

        if df_result is None or df_result.empty:
            return "A consulta não retornou nenhum resultado."

        return {
            "total_count": len(df_result),
            "sample_df": df_result.head(extract_limit_from_question(query)),
            "output_columns": relevant_columns
        }

    except Exception as e:
        print(f"❌ Erro em query_structured_data para '{file_path}': {e}")
        traceback.print_exc()
        return f"Ocorreu um erro ao processar o arquivo de dados: {e}"

def create_general_qa_tool(qa_chain):
    def query_general_company_info(query: str) -> str:
        print("Executing Faiss information query.")
        try:
            result = qa_chain.invoke({"query": query})
            return result.get("result", "Não foi possível obter uma resposta.")
        except Exception as e: return "Ocorreu um erro ao consultar a base de conhecimento."
    return query_general_company_info

def initialize_system():
    global llm, embedding_model, tools, file_manifest
    try:
        print("--- 🚀 INICIANDO SISTEMA 🚀 ---")
        with open(FILE_MANIFEST_PATH, 'r', encoding='utf-8') as f:
            file_manifest = json.load(f)
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, cache_folder=EMBEDDING_CACHE_DIR)
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
        llm = LlamaCpp(
            model_path=LLM_MODEL_PATH, temperature=0.1, n_ctx=2040, n_batch=512,
            n_threads=min(os.cpu_count() or 8, 11), verbose=False, stop=["\nCÓDIGO", "\n---", "Sua Resposta:"]
        )
        temp_tools = {}
        for item in file_manifest:
            tool_name, file_path = item["tool_name"], item["file_path"]
            if item.get("tool_type") == "vector_qa":
                vector_store = FAISS.load_local(file_path, embedding_model, allow_dangerous_deserialization=True)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, retriever=vector_store.as_retriever(search_kwargs={'k': 4}),
                    chain_type="stuff", chain_type_kwargs={"prompt": GENERAL_QA_PROMPT}
                )
                temp_tools[tool_name] = create_general_qa_tool(qa_chain)
            elif item.get("tool_type") == "dataframe_agent":
                # The function is now more flexible, but we keep the name for consistency
                temp_tools[tool_name] = lambda q, path=file_path: query_structured_data(q, path)
        tools = temp_tools
        print("--- ✨ SISTEMA PRONTO PARA USO ✨ ---")
    except Exception as e:
        print(f"❌ ERRO FATAL DURANTE A INICIALIZAÇÃO: {e}")
        traceback.print_exc()
        llm = embedding_model = None
        tools = {}

@app.route("/healthcheck")
def healthcheck():
    return jsonify({"status": "ready" if llm and tools else "initializing"}), 200 if llm and tools else 503

@app.route("/ask", methods=["POST"])
def ask_question():
    if not llm or not tools:
        return jsonify({"error": "O sistema ainda está inicializando. Por favor, aguarde."}), 503
    question = request.get_json().get("question", "").strip()
    if not question:
        return jsonify({"error": "Nenhuma pergunta fornecida."}), 400
    print(f"\n\n--- Nova Pergunta Recebida: '{question}' ---")
    
    tools_description = "\n".join([f"- tool_name: {i['tool_name']}\n  description: {i['description']}" for i in file_manifest])
    llm_choice_raw = llm.invoke(ROUTER_PROMPT.format(tools_description=tools_description, question=question))
    chosen_tool_name = next((tool for tool in tools if tool in llm_choice_raw), None) # More robust parsing
    print(f"🤖 Roteador escolheu a ferramenta: '{chosen_tool_name}'")
    
    tool_to_execute = tools.get(chosen_tool_name)
    if not tool_to_execute:
        return jsonify({"error": f"Erro interno: ferramenta '{chosen_tool_name}' não foi encontrada."}), 500
    
    final_answer = ""
    try:
        tool_result = tool_to_execute(question)
        print(f"📝 Saída Bruta da Ferramenta:\n---\n{tool_result}\n---")

        if isinstance(tool_result, str):
            final_answer = tool_result
        elif isinstance(tool_result, dict):
            total_count = tool_result.get("total_count", 0)
            sample_df = tool_result.get("sample_df")
            output_columns = tool_result.get("output_columns", [])
            
            if sample_df is None:
                final_answer = "A consulta não retornou dados para exibir. reformule sua pergunta."
            else:
                # limit = len(sample_df)
                # header = f"Encontrei **{total_count}** resultado(s)."
                # if total_count > limit: header += f" Mostrando os primeiros {limit}:"
                # else: header += ""
                
                # body_parts = []
                # <<< REFACTORED: More dynamic display column selection >>>
                # priority_cols = ['Nome', 'Projeto', 'Status', 'Cargo', 'Nome Projeto', 'Numero Projeto','Atividade'],'Formação'
                # display_cols = [col for col in priority_cols if col in output_columns]

                # if not display_cols:
                #     # Fallback to first 6 text columns if no priority cols are found
                #     display_cols = [col for col in output_columns if sample_df[col].dtype == 'object'][:6]
                # if not display_cols:
                #     display_cols = output_columns[:4] # Absolute fallback
                # display_cols = output_columns


                # for index, row in sample_df.iterrows():
                #     item_summary = " | ".join([f"**{col}:** {row[col]}" for col in display_cols if pd.notna(row[col])])
                #     body_parts.append(f"\n- {item_summary}")
                
                # final_answer = header + "".join(body_parts)


                dados_brutos_str = sample_df[output_columns].to_markdown(index=False)
                
                # Create the final prompt for the summarizer
                final_prompt = SUMMARIZER_PROMPT.format(
                    dados_brutos=dados_brutos_str,
                    question=question
                )
                
                # Invoke the LLM to get a natural language answer
                final_answer = llm.invoke(final_prompt)

    except Exception as e:
        print(f"❌ Erro ao executar a ferramenta '{chosen_tool_name}': {e}")
        final_answer = "Ocorreu um erro durante a execução da sua solicitação."

    print(f"💬 Resposta Final Programática:\n---\n{final_answer}\n---")

    def generate_response_stream():
        # Split the answer by whitespace BUT keep the delimiters (spaces, newlines)
        tokens = re.split(r'(\s+)', final_answer)
        for token in tokens:
            if token:  # Ensure not to send empty strings
                # Use json.dumps to safely handle newlines and other special characters
                yield f"data: {json.dumps(token)}\n\n"
                time.sleep(0.02)  # A short delay for a nice typing effect
        yield "data: [END]\n\n"

    return Response(generate_response_stream(), content_type="text/event-stream")

if __name__ == "__main__":
    # Ensure manifest has tool_type for proper loading
    print("Verifique se o seu 'file_manifest.json' tem 'tool_type' para cada ferramenta.")
    threading.Thread(target=initialize_system, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", port=5000)