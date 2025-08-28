import os
import pandas as pd
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

# --- Configure isso com seus dados ---
LLM_MODEL_PATH = "C:/Users/perna/Desktop/NOSTRADECIUS/gemma-3-4b-it-UD-Q5_K_XL.gguf"
TEST_QUESTION = "listar 4 empregados que trabalha e seus cargos"

# Crie um DataFrame de exemplo idêntico ao que a função teria
data = {'Nome': ['João Silva', 'Maria Oliveira', 'Pedro Souza', 'Ana Costa', 'Carlos Lima'],
        'Cargo': ['Analista de Dados', 'Engenheira de Software', 'Gerente de Projetos', 'Designer UX', 'Analista de RH']}
df_test = pd.DataFrame(data)
relevant_columns_test = ['Cargo', 'Nome']

# --- Cole o prompt exato que está falhando ---
PANDAS_CODE_GENERATOR_PROMPT = PromptTemplate.from_template(
    """Sua tarefa é gerar uma ÚNICA linha de código Python para filtrar um DataFrame pandas `df` com base em uma pergunta do usuário.

    --- INFORMAÇÕES DO DATAFRAME (Use APENAS estas colunas) ---
    Nome do DataFrame: `df`
    Colunas Disponíveis: {columns}
    Primeiras 5 linhas:
    {head}
    ----------------------------------

    --- REGRAS ABSOLUTAS ---
    1.  **A REGRA MAIS IMPORTANTE**: Use APENAS e EXCLUSIVAMENTE as colunas listadas em "Colunas Disponíveis". NUNCA, sob NENHUMA circunstância, invente um nome de coluna ou use um nome de coluna que não esteja nessa lista.
    2.  O código gerado DEVE ser uma única linha e ser atribuído a `df_result`. NÃO use `print()`.
    3.  Para filtros de texto (string), use SEMPRE a sintaxe `.str.contains('valor_da_pergunta', case=False, na=False)`.
    4.  **PERGUNTAS GERAIS**: Se a pergunta do usuário for geral e não contiver um valor específico para filtrar (ex: "liste os funcionários", "mostre os dados", "quais os cargos"), o código gerado DEVE ser `df_result = df`. A frase "que trabalha" é considerada geral e não deve ser usada para criar um filtro.
    5.  Combine múltiplos filtros usando `&` e envolvendo cada condição em parênteses `()`.
    6.  **VERIFICAÇÃO FINAL**: Antes de gerar o código, verifique se CADA nome de coluna que você usou está presente na lista de "Colunas Disponíveis". Se a pergunta não puder ser respondida com as colunas disponíveis, simplesmente não aplique filtros.

    --- PERGUNTAS DO USUÁRIO ---
    "{question}"

    --- CÓDIGO PYTHON GERADO ---
    """
)

print("--- 🚀 INICIANDO TESTE DE ISOLAMENTO 🚀 ---")
print("🔄 Carregando LLM...")
llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=0.0,
    n_ctx=4096,
    n_batch=512,
    verbose=False,
)
print("✅ LLM Carregado.")

print("\n--- FORMATANDO PROMPT PARA O TESTE ---")
final_prompt_for_test = PANDAS_CODE_GENERATOR_PROMPT.format(
    columns=relevant_columns_test,
    head=df_test[relevant_columns_test].head().to_markdown(),
    question=TEST_QUESTION
)

print("\n--- EXECUTANDO INVOKE ---")
response = llm.invoke(final_prompt_for_test)

print("\n--- ✅✅✅ RESULTADO DO MODELO ✅✅✅ ---")
print(response)
print("------------------------------------------")