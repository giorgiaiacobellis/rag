from langchain.chains import create_retrieval_chain
from datasets import Dataset

import datetime
import os
import utils
import data
import json

os.environ["OPENAI_API_KEY"] = ("sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="ragTestServer"

#VARIABILI DI CONTROLLO
SPLIT = 1  # vale zero se lo split è da eseguire altimenti 1
VECTORDB = 1 #vale 0 se il vectordb è da costruire altrimenti 1


def main():

    splits = utils.split_data(SPLIT)  # Caricamento dei dati e divisione in chunk
    retriever = utils.create_vector_db(VECTORDB,data.config, splits)  # Creazione del Vector DB

    print("chattiamo!")
    question_answer_chain = utils.generate_chat(data.config)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    answers = []
    contexts = []

    for q in data.questions:
        response = rag_chain.invoke({"input": q})
        print(response)

        answers.append(response["answer"])
        documents = [doc.page_content for doc in response["context"]]
        contexts.append(documents)

    dataset_dict = {
        "model" : data.config["llm"],
        "data" : {"question": data.questions,
                   "answer": answers,
                   "contexts": contexts,}
    }
    if data.answers is not None:
        dataset_dict["data"]["ground_truth"] = data.answers

    #save results 
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"dataset_{timestamp}.json" 
    with open(filename, "w") as outfile:
        json.dump(dataset_dict, outfile) 
    
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()



"""
0) se non è stato ancora fatto, caricare i dati nel vectordb
1) caricare il modello da testare
2) dare al modello le question, ottenere le risposte e creare un file per ogni modello che contenga risposte e ground truth
3) calcolare le metriche di valutazione per ogni modello 
4) fare il confronto tra i modelli
"""
