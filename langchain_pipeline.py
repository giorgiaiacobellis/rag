from langchain.chains import create_retrieval_chain
from datasets import Dataset

import datetime
import os
import utils
import data

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

    # Esecuzione della RAG per ogni domanda
    responses = rag_chain.batch({"input": data.questions})

    answers = []
    contexts = []

    for response in responses:
        #response = rag_chain.invoke({"input": q})

        answers.append(response["answer"])
        contexts.append(response["context"])

    dataset_dict = {
        "model" : data.config["llm"],
        "question": data.questions,
        "answer": answers,
        "contexts": contexts,
    }
    if data.answers is not None:
        dataset_dict["ground_truth"] = data.answers
    ds = Dataset.from_dict(dataset_dict)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"dataset_{timestamp}.json" 

    # Salva il dataset in un file JSON
    ds.to_json(filename)
       

if __name__ == "__main__":
    main()



"""
0) se non è stato ancora fatto, caricare i dati nel vectordb
1) caricare il modello da testare
2) dare al modello le question, ottenere le risposte e creare un file per ogni modello che contenga risposte e ground truth
3) calcolare le metriche di valutazione per ogni modello 
4) fare il confronto tra i modelli
"""
