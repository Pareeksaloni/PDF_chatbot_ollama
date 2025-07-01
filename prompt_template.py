##Define question format

from langchain.prompts import PromptTemplate

def get_prompt_template():
    template = """
    You are an assistant helping answer questions based on a document.
    Use ONLY the information from the document below.

    Document:
    {context}

    Question:
    {question}

    Answer:"""
    
    return PromptTemplate(input_variables=["context", "question"], template=template)
