# !pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
	filename="Meta-Llama-3-8B-Instruct.Q2_K.gguf",
)


llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)