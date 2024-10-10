import json
import re
import data
from langchain_community.llms.vllm import VLLM
import sys
from pysbd import Segmenter

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


def split_statements(answer):
    #print("split di: ", answer)
    #Splits the generated answer into individual statements based on punctuation and line breaks.
    sentences = sentence_segmenter.segment(answer)
    sentences = [
        sentence for sentence in sentences if sentence.strip().endswith(".")
    ]
    return sentences

def check_statement_relevance(statement, context):
    #Returns True if relevant, False otherwise.
    
    prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
              f"Dato contesto e affermazione, determina se l'affermazione può essere inferita o supportata dal contesto dato. Rispondi solamente 'si' se è supportato, rispondi solamente 'no' altrimenti. Non aggiungere altre descrizioni o frasi aggiuntive.<|eot_id|>\n"
              f"<|start_header_id|>user<|end_header_id|>\n"
              f"Data il seguente contesto:\n\n{context}\n\n"
              f"E la seguente affermazione: \n{statement}<|eot_id|>\n\n"
                "<|start_header_id|>assistant<|end_header_id|>\n Risposta: \n\n"
    )
    
    #print("Check statement: " , prompt)
    # Query the LLM to get the answer
    result = llm.invoke(prompt)
    #print ("risultato: ", result)
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
    
    #print("fine calcolo")
    # Faithfulness score is the ratio of relevant statements to total statements
    faithfulness_score = relevant_statements / total_statements
    return faithfulness_score


def faithfulness_score(filename):
    total_score = 0
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, item in enumerate(data['data']['question']):
        answer = data['data']['answer'][i]
        context = " ".join(data['data']['contexts'][i])

        # Calculate the faithfulness score
        score = calculate_faithfulness_score(context, answer)

        total_score = total_score + score
        #print(f"Faithfulness score intermedio: {score:.2f}")

    return total_score/len(data['data']['question'])


def main():
    filename = sys.argv[1]
    print(f"Test file: {filename}")
    result = faithfulness_score(filename)
    print(f"Faithfulness score totale: {result}")
    data = {"test": filename, "faithfulness_score": result}
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