##Combine model + prompt

from langchain.chains import LLMChain
from ollama_llm import get_ollama_llm
from prompt_template import get_prompt_template

def get_chain():
    llm = get_ollama_llm()
    prompt = get_prompt_template()
    return LLMChain(llm=llm, prompt=prompt)
