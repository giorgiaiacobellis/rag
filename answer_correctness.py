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
os.environ["LANGCHAIN_API_KEY"] = "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"

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


#Dato un testo generato come risposta e il testo di riferimento ("ground truth"), confronta la RISPOSTA generata con la risposta di RIFERIMENTO. Valuta se la risposta generata trasmette correttamente lo stesso significato, anche se la formulazione è diversa. Restituisci una di queste etichette: 'Corretto' o 'Errato'.
def check_statement_relevance(statement,gt):
    prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"Dato un testo generato come risposta e il riferimento ('ground truth'), confronta la RISPOSTA generata con la risposta di RIFERIMENTO."
        f"Valuta se la risposta generata trasmette correttamente lo stesso significato, anche se la formulazione è diversa o se lo è solo in parte."
        f"Restituisci una di queste etichette: 'Corretto' o 'Errato'. Non aggiungere altre informazioni.<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Ecco la risposta:\n\n{statement}<|eot_id|>\n\n"
        f"Ecco il riferimento:\n\n{gt}<|eot_id|>\n\n"
        "<|start_header_id|>assistant<|end_header_id|>Generazione:\n\n"
    )
    
    result = llm.invoke(prompt)
    return result.strip().lower() == "corretto" 


def calculate_answer_correctness_score(answer, gt):

    correctness = 0
    if check_statement_relevance(answer,gt):
        correctness
    return correctness


def answer_correctness_score(filename):
    total_score = 0
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, item in enumerate(data['data']['question']):
        answer = data['data']['answer'][i]
        gt = data['data']['ground_truth'][i]
        
        # Calculate the correctness score
        if check_statement_relevance(answer, gt):
            total_score = total_score + 1

    return total_score/len(data['data']['question'])

def main():
    filename = sys.argv[1]
    print(f"Test file: {filename}")
    result = answer_correctness_score(filename)
    print(f"Correctness score totale: {result}")
    data = {"test": filename, "answer_correctness_score": result}
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