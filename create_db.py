# from sqlalchemy import create_engine, text

# # Creates a SQLite database file named 'example.db'
# db_url = "sqlite:///TMS.db"
# engine = create_engine(db_url)

# with engine.connect() as connection:
#     # Create a sample 'employees' table
#     connection.execute(text("""
#         CREATE TABLE IF NOT EXISTS employees (
#             id INTEGER PRIMARY KEY,
#             name TEXT,
#             department TEXT,
#             salary INTEGER
#         );
#     """))

#     # Insert sample data
#     connection.execute(text("INSERT INTO employees (name, department, salary) VALUES ('Alice', 'Engineering', 90000);"))
#     connection.execute(text("INSERT INTO employees (name, department, salary) VALUES ('Bob', 'Engineering', 95000);"))
#     connection.execute(text("INSERT INTO employees (name, department, salary) VALUES ('Charlie', 'Sales', 70000);"))
#     connection.execute(text("INSERT INTO employees (name, department, salary) VALUES ('David', 'Sales', 75000);"))
#     connection.execute(text("INSERT INTO employees (name, department, salary) VALUES ('Vincent Pernarh', 'Sales', 6700);"))
#     connection.execute(text("INSERT INTO employees (name, department, salary) VALUES ('Decio Oliveira', 'Sales', 13000);"))

#     print("Database 'example.db' created and populated successfully.")
#     connection.commit()
# =========================
# LangChain SQL Agent CLI Test
# =========================

import os
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import LlamaCpp

# --- Configuration ---
# Ensure LLM_MODEL_PATH is correct
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "C:/Users/perna/Desktop/NOSTRADECIUS/openhermes-2-mistral-7b.Q4_K_M.gguf")

# =========================
# Database & Agent Setup
# =========================

# Connect to the database
db = SQLDatabase.from_uri("sqlite:///TMS.db")

# Initialize the LlamaCpp LLM
llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=0.0,
    n_ctx=4096,
    n_batch=512,
    n_threads=min(os.cpu_count() or 8, 11),
    verbose=False,
    # stop=["<|im_end|>"]
)

# Create the SQL Database Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,  # Set to True to see the agent's thought process
    handle_parsing_errors=True
)


# =========================
# Main execution block
# =========================

if __name__ == '__main__':
    print("Welcome to the SQL Agent CLI test environment!")
    print("The agent is connected to the 'TMS.db' database and is ready to answer questions.")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 50)

    while True:
        question = input("Your query: ")

        if question.lower() in ["exit", "quit"]:
            print("Exiting session. Goodbye!")
            break

        if not question.strip():
            print("Please enter a valid query.")
            continue

        try:
            # Run the agent with the user's question
            response = agent_executor.invoke({"input": question})

            # Print the final answer from the agent
            print("\nAgent's response:")
            print(response.get('output', "No response found."))
            print("-" * 50)

        except Exception as e:
            print(f"An error occurred: {e}")
            print("-" * 50)

