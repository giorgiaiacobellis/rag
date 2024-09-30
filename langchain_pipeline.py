from langchain.chains import create_retrieval_chain
from datasets import Dataset

import datetime
import os
import utils
import data
import json


from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator


os.environ["OPENAI_API_KEY"] = ("sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="ragTestServer"

#VARIABILI DI CONTROLLO
SPLIT = 1  # vale zero se lo split è da eseguire altimenti 1
VECTORDB = 1 #vale 0 se il vectordb è da costruire altrimenti 1

def create_evaluator(data,evaluator,predict_rag_answer):
    client = Client()
    dataset_name = "rag_dataset"
    dataset = client.create_dataset(dataset_name=dataset_name, description="Dataset for RAG evaluation")
    client.create_examples(inputs=[{"question": q} for q in data["data"]["question"]],
                        outputs=[{"ground_truth": a} for a in data["data"]["ground_truth"]],
                        dataset_id=dataset.id)

    qa_evaluator = LangChainStringEvaluator("cot_qa",
                                            config={"llm": evaluator},
                                            prepare_data=lambda run, example: {
                                                "prediction": run.outputs["answer"],
                                                "reference": example.outputs["ground_truth"],
                                                "input": example.inputs["question"],
                                            },) 
    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=qa_evaluator,
        experiment_prefix="rag-qa-oai",
    )
    return experiment_results

def main():

    splits = utils.split_data(SPLIT)  # Caricamento dei dati e divisione in chunk
    retriever = utils.create_vector_db(VECTORDB,data.config, splits)  # Creazione del Vector DB

    print("chattiamo!")
    question_answer_chain, evaluator = utils.generate_chat(data.config)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # RAG chain
    def predict_rag_answer(example: dict):
        """Use this for answer evaluation"""
        response = rag_chain.invoke({"input": example["question"]})
        return {"answer": response["answer"]}

    def predict_rag_answer_with_context(example: dict):
        """Use this for evaluation of retrieved documents and hallucinations"""
        response = rag_chain.invoke({"input": example["question"]})
        return {"answer": response["answer"], "contexts": response["contexts"]}


    dataset_dict = {
        "model" : data.config["llm"],
        "data" : {"question": data.questions,
                  "ground_truth": data.ground_truth,
                  }
    }

    eval_results = create_evaluator(dataset_dict,evaluator,predict_rag_answer)
    print(eval_results)


if __name__ == "__main__":
    main()



"""
0) se non è stato ancora fatto, caricare i dati nel vectordb
1) caricare il modello da testare
2) dare al modello le question, ottenere le risposte e creare un file per ogni modello che contenga risposte e ground truth
3) calcolare le metriche di valutazione per ogni modello 
4) fare il confronto tra i modelli
"""
