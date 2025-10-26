import os
import json
import time
import threading
import re
import pandas as pd
import traceback
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain_community.chat_message_histories import ChatMessageHistory
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
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "C:/Users/perna/Desktop/NOSTRADECIUS/openhermes-2-mistral-7b.Q4_K_M.gguf")
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
Você é o Nostradecius, um assistente de IA factual. Sua única função é responder à pergunta do usuário usando exclusivamente o CONTEXTO fornecido.

REGRAS ESTRITAS:
1. Responda em Português do Brasil.
2. Se a informação para responder à pergunta não estiver no CONTEXTO, responda EXATAMENTE: 'Não encontrei dados relevantes sobre isso.'
3. Baseie-se APENAS no CONTEXTO. Não use conhecimento externo ou faça suposições.<|im_end|>
<|im_start|>user
CONTEXTO:
{context}

PERGUNTA DO USUÁRIO: {question}<|im_end|>
<|im_start|>assistant
RESPOSTA:
"""
)

CODE_GENERATOR_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é agir como um especialista em pandas Python. Dada uma pergunta do usuário e uma lista de colunas COM SEUS TIPOS DE DADOS, seu objetivo é gerar uma única linha de código Python que filtra e seleciona dados de um DataFrame chamado `df`.

REGRAS CRÍTICAS:
1.  Gere APENAS uma linha de código.
2.  **DATAS (`datetime64[ns]`):** Para filtrar datas, **SEMPRE** use o acessador `.dt`. Por exemplo, para filtrar o ano, use `df['NomeDaColunaData'].dt.year == 2025`. **NUNCA** use `.astype(str).str.contains()` em colunas de data, pois é impreciso. O sistema já garante que essas colunas estão no formato de data correto.
3.  **TEXTO (`object`):** Para colunas de texto, use `df['NomeDaColunaTexto'].str.contains('valor', case=False, na=False)`.
4.  **SELEÇÃO DE COLUNAS:** Após a filtragem, selecione APENAS as colunas que o usuário pediu para ver.

EXEMPLO 1 (Filtro de Texto):
Pergunta: "Liste 3 projetos com status 'Em Andamento'"
Colunas e Tipos:
Nome Projeto     object
Nome Analista    object
Status           object
Código Gerado:
df_result = df[df['Status'].str.contains('Em Andamento', case=False, na=False)][['Nome Projeto']].head(3)

EXEMPLO 2 (Filtro de Data):
Pergunta: "me mostre os projetos criados em 2024"
Colunas e Tipos:
Nome Projeto        object
Data Criação   datetime64[ns]
Status              object
Código Gerado:
df_result = df[df['Data Criação'].dt.year == 2024][['Nome Projeto']]<|im_end|>
<|im_start|>user
Pergunta: "{question}"
Colunas e Tipos:
{columns_and_types}

Gere o código Python de uma linha.<|im_end|>
<|im_start|>assistant
"""
)

SUMMARIZER_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é atuar como um assistente de IA que resume os dados encontrados de forma clara e natural em Português.

REGRAS CRÍTICAS:
1.  Baseie sua resposta ESTRITAMENTE nos dados fornecidos: `PERGUNTA ORIGINAL DO USUÁRIO`, `CONTAGEM TOTAL REAL`, e `AMOSTRA DE DADOS`. Não invente informações.
2.  Comece com uma única frase que resume o que foi encontrado, conectando-se diretamente à pergunta do usuário.
3.  Após a frase de resumo, introduza a lista de dados de forma precisa. Por exemplo:
    - Se o usuário pediu um número específico (ex: "liste 4") e a CONTAGEM TOTAL é maior, diga "Aqui estão os 4 itens solicitados de um total de X encontrados:".
    - Se o usuário pediu um número e a CONTAGEM TOTAL é menor ou igual ao pedido, diga "Encontrei X itens que correspondem à sua busca:".
    - Se a CONTAGEM TOTAL e o tamanho da AMOSTRA são iguais, diga "Aqui estão todos os X resultados encontrados:".
4.  Use Markdown para formatar a resposta. Destaque os principais termos da pergunta do usuário em negrito (`**palavra**`).

EXEMPLO DE FLUXO COMPLETO:
- PERGUNTA DO USUÁRIO: "Liste 4 projetos do analista Decio com status On Hold"
- DADOS: CONTAGEM TOTAL REAL = 6, AMOSTRA DE DADOS tem 4 projetos.
- RESPOSTA IDEAL GERADA:
"Sua busca por **projetos** do analista **Decio** com status **On Hold** encontrou um total de **6** resultados. Aqui estão os 4 solicitados:
[Tabela ou Lista com os 4 projetos]"
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
# Query Engine
# =========================
def query_structured_data(query: str, file_path: str):
    print(f"Executing intelligent structured data query on: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        # This pre-processing step is crucial - keep it!
        for col in df.columns:
            if 'data' in col.lower() or 'date' in col.lower():
                try: 
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception: 
                    pass

        print("🧠 Etapa 1/1: Gerando código de filtragem com LLM...")
        
        column_info = df.dtypes.to_string()
        code_gen_prompt = CODE_GENERATOR_PROMPT.format(
            columns_and_types=column_info, 
            question=query
        )
        generated_code = llm.invoke(code_gen_prompt).strip()

        if generated_code.startswith("```python"):
            generated_code = generated_code[9:]
        if generated_code.startswith("```"):
            generated_code = generated_code[3:]
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3]
        generated_code = generated_code.strip()
        if not generated_code.strip().startswith("df_result"):
            generated_code = "df_result = " + generated_code

        print(f"✅ Código Gerado:\n---\n{generated_code}\n---")
        
        # --- NEW, SIMPLIFIED & ROBUST LOGIC ---

        # Execute the generated code just ONCE
        scope = {"df": df, "pd": pd}
        exec(generated_code, {}, scope)
        result = scope.get('df_result')

        # CASE 1: The result is a single number (from a .count(), .sum(), etc. query)
        if isinstance(result, (int, float)) or 'numpy' in str(type(result)):
            # The query was a direct calculation. We return a dictionary so the
            # main loop can use the Summarizer LLM to formulate a natural response.
            return {
                "total_count": int(result),
                "sample_df": pd.DataFrame() # Return an empty DataFrame
            }

        # CASE 2: The result is a DataFrame or a Series (a standard query for data)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            df_sample_result = result
            
            if isinstance(df_sample_result, pd.Series):
                df_sample_result = df_sample_result.to_frame()

            if df_sample_result.empty:
                return "A consulta não retornou nenhum resultado."

            # Now, calculate the TRUE total count by removing any .head() limit from the code
            code_for_counting = re.sub(r'\.head\(\s*\d+\s*\)\s*$', '', generated_code)
            scope_count = {"df": df, "pd": pd}
            exec(code_for_counting, {}, scope_count)
            df_full_result = scope_count.get('df_result')
            total_count = len(df_full_result) if isinstance(df_full_result, (pd.DataFrame, pd.Series)) else 0

            return {
                "total_count": total_count,
                "sample_df": df_sample_result,
                "output_columns": df_sample_result.columns.tolist(),
                "generated_code": generated_code
            }
        
        # Fallback if the generated code produces an unexpected type
        return "A consulta não retornou um formato de resultado esperado."

    except Exception as e:
        print(f"❌ Erro em query_structured_data para '{file_path}': {e}")
        traceback.print_exc()
        return f"Ocorreu um erro ao processar o arquivo de dados: {e}"
# =========================
# Tool and System Initialization
# =========================
def create_general_qa_tool(qa_chain):
    canned_answers = {
        "quem te desenvolveu": "Eu sou o Nostradecius, um assistente de IA factual desenvolvido por Vincent Pernarh, Vitoria Andrade e Decio Martins.",
        "qual o seu nome": "Meu nome é Nostradecius, seu assistente de IA.",
        "sobre voce": "Eu sou o Nostradecius, um assistente de IA factual. Minha função é responder perguntas usando a base de conhecimento interna. Fui desenvolvido por Vincent Pernarh, Vitoria Andrade e Decio Martins.",
        "sobre você": "Eu sou o Nostradecius, um assistente de IA factual. Minha função é responder perguntas usando a base de conhecimento interna. Fui desenvolvido por Vincent Pernarh, Vitoria Andrade e Decio Martins.",
        "o que você faz": "Minha função é atuar como um assistente de IA para responder perguntas, consultar dados e fornecer informações com base nos documentos e bases de dados da empresa.",
        "o que voce faz": "Minha função é atuar como um assistente de IA para responder perguntas, consultar dados e fornecer informações com base nos documentos e bases de dados da empresa. DHL"
    }
    
    def query_general_company_info(query: str) -> str:
        print("Executing Faiss information query.")
        
        query_lower = query.lower().strip()
        for key, answer in canned_answers.items():
            if key in query_lower:
                return answer

        try:
            result = qa_chain.invoke({"query": query})
            return result.get("result", "Não foi possível obter uma resposta.")
        except Exception as e:
            print(f"Error during FAISS query: {e}")
            return "Ocorreu um erro ao consultar a base de conhecimento."
            
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
            model_path=LLM_MODEL_PATH,
            temperature=0.0,
            n_ctx=4096,
            n_batch=512,
            n_threads=min(os.cpu_count() or 8, 11),
            verbose=False,
            stop=["<|im_end|>"]
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
                temp_tools[tool_name] = lambda q, path=file_path: query_structured_data(q, path)
        tools = temp_tools
        print("--- ✨ SISTEMA PRONTO PARA USO ✨ ---")
    except Exception as e:
        print(f"❌ ERRO FATAL DURANTE A INICIALIZAÇÃO: {e}")
        traceback.print_exc()
        llm = embedding_model = None
        tools = {}

# =========================
# Flask Routes
# =========================
@app.route("/healthcheck")
def healthcheck():
    return jsonify({"status": "ready" if llm and tools else "initializing"}), 200 if llm and tools else 503

@app.route("/ask", methods=["POST"])
def ask_question():
    if not llm or not tools:
        return jsonify({"error": "O sistema ainda está inicializando. Por favor, aguarde."}), 503
    
    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id")

    if not question:
        return jsonify({"error": "Nenhuma pergunta fornecida."}), 400
    if not session_id:
        return jsonify({"error": "Nenhum ID de sessão ('session_id') foi fornecido."}), 400

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
                tools_description=tools_description, 
                question=question,
                chat_history=chat_history
            )
            llm_choice_raw = llm.invoke(router_prompt_formatted)
            chosen_tool_name = next((tool for tool in tools if tool in llm_choice_raw), None)
            print(f"🤖 Roteador escolheu a ferramenta (com contexto): '{chosen_tool_name}'")
            
            tool_to_execute = tools.get(chosen_tool_name)
            if not tool_to_execute:
                final_answer = f"Erro interno: ferramenta '{chosen_tool_name}' não foi encontrada."
            else:
                tool_result = tool_to_execute(question)
                print(f"📝 Saída Bruta da Ferramenta:\n---\n{tool_result}\n---")

                # =========================================================================
                # --- UPDATED LOGIC to properly use the summarizer for all cases ---
                # =========================================================================
                if isinstance(tool_result, str):
                    final_answer = tool_result
                elif isinstance(tool_result, dict):
                    # This block now handles both queries that return data and those that only return a count.
                    sample_df = tool_result.get("sample_df", pd.DataFrame())
                    total_count = tool_result.get("total_count", 0)
                    
                    # Default to an empty string for the raw data. If the dataframe isn't empty, fill it.
                    dados_brutos_str = ""
                    if sample_df is not None and not sample_df.empty:
                        dados_brutos_str = sample_df.to_markdown(index=False)

                    # Always call the summarizer. It's designed to create a good response
                    # even if it only receives a total_count and the original question.
                    final_prompt = SUMMARIZER_PROMPT.format(
                        dados_brutos=dados_brutos_str,
                        question=question,
                        total_count=total_count
                    )
                    final_answer = llm.invoke(final_prompt)
                else:
                    final_answer = "A ferramenta retornou um resultado em formato inesperado."
                # =========================================================================
    
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