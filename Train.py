import pandas as pd
import re
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

# --- Configurações ---
CSV_FILE_PATH = "C:/Users/perna/Desktop/NOSTRADECIUS/training_data.csv"
FAISS_INDEX_PATH = "company_faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Diretório de cache para o modelo
CACHE_DIR = "models_cache"


def clean_text(text: str) -> str:
    """Limpa uma string removendo caracteres inválidos e espaços extras."""
    if not isinstance(text, str):
        text = str(text)
    try:
        text = re.sub(r'\s+', ' ', text).strip()
        text = ''.join(char for char in text if char.isprintable())
        return text
    except Exception:
        return ""


def load_and_format_csv_data(file_path: str) -> List[str]:
    """Carrega dados de um CSV e formata cada linha em uma string estruturada."""
    if not os.path.exists(file_path):
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return []

    try:
        df = pd.read_csv(file_path, header=0, dtype=str).fillna("")

        if df.empty:
            print(f"⚠️  O arquivo está vazio: {file_path}")
            return []

        formatted_documents = []
        for _, row in df.iterrows():
            parts = [f"{clean_text(col)}: {clean_text(row[col])}" 
                     for col in df.columns if clean_text(row[col])]
            row_text = ". ".join(parts)
            
            if row_text:
                formatted_documents.append(row_text)

        print(f"✅ Carregados e formatados {len(formatted_documents)} documentos de {file_path}")
        return formatted_documents

    except Exception as e:
        print(f"❌ Erro ao processar o arquivo CSV {file_path}: {e}")
        return []


def create_faiss_index():
    """Função principal para carregar dados, gerar embeddings e salvar o índice FAISS."""
    print("🚀 Iniciando o processamento de dados...")

    documents = load_and_format_csv_data(CSV_FILE_PATH)

    if not documents:
        print("❌ Nenhum documento foi carregado. Verifique o CSV_FILE_PATH. Abortando.")
        return

    try:
        print(f"🔄 Inicializando o modelo de embedding: {EMBEDDING_MODEL_NAME}...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},   # CPU para evitar problemas de GPU
            cache_folder=CACHE_DIR            # ✅ Correção: só aqui
        )
        print("✅ Modelo de embedding carregado.")

        print("🔄 Construindo o índice vetorial FAISS... (isso pode levar um momento)")
        vector_store = FAISS.from_texts(texts=documents, embedding=embedding_model)
        print("✅ Índice vetorial construído com sucesso.")

        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"💾 Índice FAISS salvo com sucesso na pasta: '{FAISS_INDEX_PATH}'")

    except Exception as e:
        print(f"❌ Ocorreu um erro durante a criação do índice FAISS: {e}")


if __name__ == "__main__":
    create_faiss_index()
