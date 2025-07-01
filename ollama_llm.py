##Connect to Ollama (local model)

from langchain_community.llms import Ollama

def get_ollama_llm(model="tinyllama"):
    return Ollama(model=model)

def process_numbers(nums: list[int]) -> int:
    return sum(nums)
