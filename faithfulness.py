import json
import re
import data
from langchain import PromptTemplate
from langchain_community.llms.vllm import VLLM

# Initialize the LLM engine using vLLM and LangChain

from pysbd import Segmenter

# Use a Mistral model - replace with the local path or Mistral model you are using
llm = VLLM(
            model=data.config_eval["llm"]["model"],
            top_p=data.config_eval["llm"]["top_p"],
            max_new_tokens=4000,
            temperature=data.config_eval["llm"]["temperature"],
            top_k=data.config_eval["llm"]["top_k"],
            trust_remote_code= True
        )
# Define a prompt for verifying statement relevance
prompt_template = """
Dato il seguente contesto:

\n{context}\n

And the following statement:

{statement}

Determine whether the statement can be inferred or supported by the context. Answer "yes" if it is supported and "no" if it is not.

Answer:
"""
# Template for the LLMChain
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "statement"])


sentence_segmenter= Segmenter(
        language="it",
        clean=False,
        char_span=False,
    )


def split_statements(answer):
    #Splits the generated answer into individual statements based on punctuation and line breaks.
    sentences = sentence_segmenter.segment(answer)
    sentences = [
        sentence for sentence in sentences if sentence.strip().endswith(".")
    ]

    # This regex assumes the statements are separated by punctuation marks like periods or newlines.
    #statements = re.split(r'(?<=[.!?])\s+', answer.strip())
    return sentences

def check_statement_relevance(statement, context):
    #Uses the LLM to determine if a statement is relevant to the context.
    #Returns True if relevant, False otherwise.
    
    prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"Data il seguente contesto:\n\n{context}\n\n"
        f"E la seguente affermazione: \n{statement}\n"
        f"determina se l'affermazione può essere inferita o supportata dal contesto dato. Rispondi 'si' se è supportato, rispondi 'no' altrimenti.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n Risposta: \n\n"
    )
    
    # Query the LLM to get the answer
    result = llm.invoke(prompt)
    
    # Check if the LLM answered "yes" or "no"
    return result.strip().lower() == "si"

def calculate_faithfulness_score(context, answer):
    #Calculates the faithfulness score for the generated answer based on the context.

    statements = split_statements(answer)
    total_statements = len(statements)
    if total_statements == 0:
        return 0  # To avoid division by zero

    relevant_statements = 0
    for statement in statements:
        if check_statement_relevance(statement, context):
            relevant_statements += 1
    
    # Faithfulness score is the ratio of relevant statements to total statements
    faithfulness_score = relevant_statements / total_statements
    return faithfulness_score

# Example usage:
if __name__ == "__main__":
    # Define the context and generated answer
    
    with open("dataset_prova.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    total_sim = 0
    for i, item in enumerate(data['data']['question']):
        original_question = item
        answer = data['data']['answer'][i]
        context = data['data']['contexts'][0][i]
        
    # Calculate the faithfulness score
    score = calculate_faithfulness_score(context, answer)
    print(f"Faithfulness score: {score:.2f}")
