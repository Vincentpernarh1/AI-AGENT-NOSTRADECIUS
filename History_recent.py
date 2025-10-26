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

# =========================
# Prompts (Updated for ChatML)
# =========================

ROUTER_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é agir como um roteador de IA. Com base na pergunta do usuário, selecione a melhor ferramenta da lista para respondê-la. Responda APENAS com o nome da ferramenta (`tool_name`). Não adicione nenhuma outra palavra ou explicação.<|im_end|>
<|im_start|>user
--- Ferramentas Disponíveis ---
{tools_description}
--------------------------------

Pergunta do Usuário: "{question}"<|im_end|>
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
Sua tarefa é agir como um especialista em pandas Python. Dada uma pergunta do usuário e uma lista de colunas, seu objetivo é gerar uma única linha de código Python que filtra e seleciona dados de um DataFrame chamado `df`.

REGRAS:
1.  Gere APENAS uma linha de código.
2.  **FILTRAGEM:** Use `df[...]` com as condições de filtro.
3.  **SELEÇÃO DE COLUNAS:** Após a filtragem, selecione APENAS as colunas que o usuário pediu para ver.
4.  **LIMITE DE LINHAS:** Se o usuário pedir um número específico de itens (ex: "liste 4 projetos"), adicione `.head(4)` no final.
5.  **PERGUNTAS GERAIS:** Ignore frases conversacionais como "que você conhece", "me mostre", etc. Se NENHUMA entidade específica (nome, status, data) for encontrada na pergunta para usar como filtro, gere um código que seleciona colunas e aplica um limite, sem filtrar.

EXEMPLO 1 (Com Filtros):
Pergunta: "Liste os nomes de 3 projetos para o analista João Silva com status 'Em Andamento'"
Colunas: ['Nome Projeto', 'Nome Analista', 'Status', 'Data de Criação']
Código Gerado:
df_result = df[(df['Nome Analista'].str.contains('João Silva', case=False, na=False)) & (df['Status'].str.contains('Em Andamento', case=False, na=False))][['Nome Projeto']].head(3)

EXEMPLO 2 (Sem Filtros):
Pergunta: "me mostre 3 funcionários e seus cargos"
Colunas: ['Nome', 'Cargo', 'Atividade']
Código Gerado:
df_result = df[['Nome', 'Cargo']].head(3)<|im_end|>
<|im_start|>user
Pergunta: "{question}"
Colunas: {columns}

Gere o código Python de uma linha.<|im_end|>
<|im_start|>assistant
"""
)

COLUMN_SELECTOR_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é selecionar todas as colunas de uma lista que possam ser necessárias ou relevantes para responder à pergunta de um usuário, considerando relações diretas e indiretas.

INSTRUÇÕES ADICIONAIS:
- Para cada valor, nome, status, data ou entidade mencionada na pergunta, inclua qualquer coluna cujo nome possivelmente armazene essa informação (ex: se a pergunta mencionar um analista, inclua colunas como 'Nome Analista', 'Analista', etc).
- Inclua sempre as colunas necessárias para filtrar, identificar ou responder partes específicas da pergunta.
- Se houver dúvida, seja inclusivo: prefira selecionar mais colunas do que menos.

Responda apenas com uma lista Python contendo todos os nomes das colunas selecionadas. Não adicione explicações, apenas a lista.<|im_end|>
<|im_start|>user
Pergunta do usuário: "{question}"

Colunas disponíveis: {columns}

Selecione as colunas relevantes.<|im_end|>
<|im_start|>assistant
"""
)

PROMPT_PREENCHER_LACUNAS = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é completar um objeto JSON extraindo valores EXCLUSIVAMENTE da pergunta do usuário.

REGRA CRÍTICA: Se um valor para uma chave NÃO estiver EXPLICITAMENTE na Pergunta do Usuário, você DEVE deixar o valor como uma string vazia `""`. NÃO invente, adivinhe ou presuma valores.

Responda APENAS com o objeto JSON completo.<|im_end|>
<|im_start|>user
--- Exemplo ---
Pergunta do Usuário: "mostre-me itens para Jane Doe que estão ativos"
JSON com Lacunas:
{{
  "alguma_coluna_de_nome": "",
  "alguma_coluna_de_status": "",
  "alguma_coluna_de_data": ""
}}
JSON Completo Esperado:
{{
  "alguma_coluna_de_nome": "Jane Doe",
  "alguma_coluna_de_status": "ativo",
  "alguma_coluna_de_data": ""
}}
---

**Tarefa Atual:**
Pergunta do Usuário: "{question}"

JSON com Lacunas:
{json_blanks}

Complete o JSON com base na pergunta.<|im_end|>
<|im_start|>assistant
"""
)

# --- MODIFIED PROMPT ---
SUMMARIZER_PROMPT = PromptTemplate.from_template(
    """<|im_start|>system
Sua tarefa é apresentar os resultados encontrados para o usuário de forma clara e natural em Português, combinando a contagem total com a lista de amostra.

REGRAS CRÍTICAS:
1.  Sua resposta DEVE ser baseada ESTRITAMENTE nos dados fornecidos.
2.  **Sempre** comece a resposta mencionando a contagem total de itens. Por exemplo: "O analista **Decio** tem um total de **12** projetos com status **Done**."
3.  Depois de declarar o total, se a pergunta pediu para listar itens, apresente a lista de amostra. Diga algo como "Aqui estão os 5 projetos solicitados:".
4.  Se a contagem total for igual ao número de itens na lista de amostra, diga "Aqui estão todos os X resultados encontrados:".
5.  Use Markdown para formatar a lista ou tabela. Destaque palavras-chave importantes em negrito (`**palavra**`).<|im_end|>
<|im_start|>user
PERGUNTA ORIGINAL DO USUÁRIO:
"{question}"

CONTAGEM TOTAL REAL: {total_count}

AMOSTRA DE DADOS EXIBIDA (pode ser limitada):
{dados_brutos}

Com base na pergunta, na contagem total e nos dados da amostra, apresente a resposta final.<|im_end|>
<|im_start|>assistant
"""
)


# =========================
# Utility Functions
# =========================

def extract_limit_from_question(question):
    match = re.search(r'(listar|mostre|mostra|apenas|somente|exibir|exiba|mostrar|mínimo|minimo)\s+(\d+)', question, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return 20

def get_best_match(user_value, unique_values):
    user_value_clean = str(user_value).strip().lower()
    clean_values = [str(v).strip().lower() for v in unique_values if pd.notnull(v)]
    matches = difflib.get_close_matches(user_value_clean, clean_values, n=1, cutoff=0.6)
    if matches:
        idx = clean_values.index(matches[0])
        return str(unique_values[idx])
    return user_value

# =========================
# Query Engine
# =========================
# --- MODIFIED FUNCTION ---
def query_structured_data(query: str, file_path: str):
    print(f"Executing intelligent structured data query on: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        for col in df.columns:
            if 'data' in col.lower() or 'date' in col.lower():
                try: df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception: pass

        print("🧠 Etapa 1/1: Gerando código de filtragem com LLM...")
        
        output_columns = df.columns.tolist()
        
        code_gen_prompt = CODE_GENERATOR_PROMPT.format(
            columns=output_columns,
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
        
        # --- NEW LOGIC TO SEPARATE COUNTING AND SAMPLING ---

        # 1. Create a version of the code for counting (without .head())
        code_for_counting = re.sub(r'\.head\(\s*\d+\s*\)\s*$', '', generated_code)

        # 2. Execute the counting code
        scope_count = {"df": df, "pd": pd}
        exec(code_for_counting, {}, scope_count)
        df_full_result = scope_count.get('df_result')
        
        total_count = len(df_full_result) if df_full_result is not None else 0

        # 3. Execute the original code to get the sample for display
        scope_sample = {"df": df, "pd": pd}
        exec(generated_code, {}, scope_sample)
        df_sample_result = scope_sample.get('df_result')

        if df_sample_result is None or df_sample_result.empty:
            return "A consulta não retornou nenhum resultado."

        # 4. Return the correct total count and the limited sample
        return {
            "total_count": total_count,
            "sample_df": df_sample_result,
            "output_columns": df_sample_result.columns.tolist(),
            "generated_code": generated_code
        }

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
        "sobre voce ou vc": "Eu sou o Nostradecius, um assistente de IA factual. Minha função é responder perguntas usando a base de conhecimento interna. Fui desenvolvido por Vincent Pernarh, Vitoria Andrade e Decio Martins."
    }
    
    def query_general_company_info(query: str) -> str:
        print("Executing Faiss information query.")
        
        query_lower = query.lower()
        for key, answer in canned_answers.items():
            if key in query_lower:
                return answer

        try:
            result = qa_chain.invoke({"query": query})
            return result.get("result", "Não foi possível obter uma resposta.")
        except Exception as e:
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
    question = request.get_json().get("question", "").strip()
    if not question:
        return jsonify({"error": "Nenhuma pergunta fornecida."}), 400
    print(f"\n\n--- Nova Pergunta Recebida: '{question}' ---")
    
    tools_description = "\n".join([f"- tool_name: {i['tool_name']}\n  description: {i['description']}" for i in file_manifest])
    llm_choice_raw = llm.invoke(ROUTER_PROMPT.format(tools_description=tools_description, question=question))
    chosen_tool_name = next((tool for tool in tools if tool in llm_choice_raw), None)
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
            sample_df = tool_result.get("sample_df")
            # --- MODIFIED LOGIC ---
            total_count = tool_result.get("total_count", 0)
            
            if sample_df is None or sample_df.empty:
                final_answer = "A consulta não retornou dados para exibir. reformule sua pergunta."
            else:
                dados_brutos_str = sample_df.to_markdown(index=False)
                
                final_prompt = SUMMARIZER_PROMPT.format(
                    dados_brutos=dados_brutos_str,
                    question=question,
                    total_count=total_count # Pass the real total count
                )
                final_answer = llm.invoke(final_prompt)
    except Exception as e:
        print(f"❌ Erro ao executar a ferramenta '{chosen_tool_name}': {e}")
        traceback.print_exc()
        final_answer = "Ocorreu um erro durante a execução da sua solicitação."

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