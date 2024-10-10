from langchain.chains.retrieval import create_retrieval_chain
from datasets import Dataset
import sys


import os
import utils
import data
import json

os.environ["OPENAI_API_KEY"] = ("sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")
os.environ["LANGCHAIN_API_KEY"] = "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"


#VARIABILI DI CONTROLLO
SPLIT = 1  # vale zero se lo split è da eseguire altimenti 1
VECTORDB = 1 #vale 0 se il vectordb è da costruire altrimenti 1


def generate_db(rag_chain, filename):
    
    answers = []
    contexts = []

    for q in data.questions:
        response = rag_chain.invoke({"input": q})

        answers.append(response["answer"])
        documents = [doc.page_content for doc in response["context"]]
        contexts.append(documents)

    dataset_dict = {
        "model" : data.config["llm"],
        "data" : {"question": data.questions,
                  "ground_truth": data.ground_truth,
                  "answer": answers,
                  "contexts": contexts,
                }
    }
    
    #ds  = Dataset.from_dict(dataset_dict["data"])
    #save results 
    
    #filename = data.config["filename"]
    with open(filename, "w") as outfile:
        json.dump(dataset_dict, outfile) 
    
    print(f"Results saved to {filename}")

    #return ds


def main():
    filename = sys.argv[1]
    print(f"Test file: {filename}")
    splits = utils.split_data(SPLIT)  # Caricamento dei dati e divisione in chunk
    retriever = utils.create_vector_db(VECTORDB,data.config, splits)  # Creazione del Vector DB

    print("chattiamo!")
    question_answer_chain= utils.generate_chat(data.config)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    generate_db(rag_chain, filename)


if __name__ == "__main__":
    main()



"""
0) se non è stato ancora fatto, caricare i dati nel vectordb
1) caricare il modello da testare
2) dare al modello le question, ottenere le risposte e creare un file per ogni modello che contenga risposte e ground truth
"""
