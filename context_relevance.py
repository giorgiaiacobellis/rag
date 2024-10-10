import os
import data
import json
import sys
from pysbd import Segmenter

from langchain_community.llms.vllm import VLLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


os.environ["OPENAI_API_KEY"] = ("sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="ragcontextRel"


llm = VLLM(
            model=data.config_eval["llm"]["model"],
            top_p=data.config_eval["llm"]["top_p"], #1
            max_new_tokens=4000,
            temperature=data.config_eval["llm"]["temperature"], #0.3
            top_k=data.config_eval["llm"]["top_k"], #10
            trust_remote_code= True
        )

sentence_segmenter= Segmenter(
        language="it",
        clean=False,
        char_span=False,
    )


def split_statements(context):
    sentences = sentence_segmenter.segment(context)
    sentences = [
        sentence for sentence in sentences if sentence.strip().endswith(".")
    ]
    return sentences

def check_statement_relevance(question,statement):
    prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"Calcola l'importanza dell'affermazione recuperata dal contesto nel rispondere alla domanda."
        f"Data l'affermazione e la domanda definisci quanto è utile l'affermazione per rispondere alla domanda, anche se parzialmente."
        f"Restituisci unicamente 'Rilevante' se l'affermazione è utile, 'Irrilevante' altrimenti. Non aggiungere altre informazioni.<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Ecco la domanda:\n\n{question}<|eot_id|>\n\n"
        f"Ecco l'affermazione:\n\n{statement}<|eot_id|>\n\n"
        "<|start_header_id|>assistant<|end_header_id|>Generazione:\n\n"
    )
    
    result = llm.invoke(prompt)
    return result.strip().lower() == "rilevante" 


def calculate_context_relevance_score(context, question):

    statements = split_statements(context)
    total_statements = len(statements)
    if total_statements == 0:
        return 0  

    relevant_statements = 0
    for statement in statements:
        if check_statement_relevance(question,statement):
            relevant_statements += 1
    
    relevance = relevant_statements/total_statements
    print("relevance",relevance)
    return relevance


def context_relevance_score(filename):
    total_score = 0
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, item in enumerate(data['data']['question']):
        question = item
        context = " ".join(data['data']['contexts'][i])

        # Calculate the faithfulness score
        score = calculate_context_relevance_score(context, question)

        total_score = total_score + score
        #print(f"Faithfulness score intermedio: {score:.2f}")

    return total_score/len(data['data']['question'])

def main():
    filename = sys.argv[1]
    print(f"Test file: {filename}")
    result = context_relevance_score(filename)
    print(f"Context relevancy score totale: {result}")
    data = {"test": filename, "context_relevance_score": result}
    try:
        with open("metrics_scores.json", "r") as f:
            js = json.load(f)
    except FileNotFoundError:
        js = [] 

    js.append(data)

    with open("metrics_scores.json", "w") as f:
        json.dump(js, f, indent=4)

if __name__ == "__main__":
    main()