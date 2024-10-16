from langchain_huggingface import HuggingFaceEmbeddings
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
import json
import os
import sys

from ragas.metrics import answer_similarity
from ragas import evaluate
from langchain_community.llms.vllm import VLLM
from ragas.llms import LangchainLLMWrapper


os.environ["OPENAI_API_KEY"] = ("sk-proj-aVtM"+"47AoCct6IkxYRd7kUdwAQSCO8DmcZ2Ht88YqVKmOsRUFGk6fT0aVQeAdT4M9j8aDv8pFiUT3BlbkFJ4EjTuuQ"+"_MbmTWCik4mPzQ9f8YKQcMWlcBu1boiwsnLSqPSDB2HGjuwbUDbUPx6lBFK85uYElkA")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")
os.environ["LANGCHAIN_API_KEY"] = ("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")


def create_samples_from_dataset(dataset):
    samples = []
    for question, answer, contexts, ground_truth in zip(
        dataset["question"], dataset["answer"], dataset["contexts"], dataset["ground_truth"]
    ):
        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=ground_truth
            )
        )
    return samples

#-----modelli----
evaluator = VLLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            trust_remote_code= True,
            max_new_tokens=2000,
            temperature=1, #verificare
            
        )

evaluator_llm = LangchainLLMWrapper(evaluator)

hf = HuggingFaceEmbeddings(
    model_name="dunzhang/stella_en_400M_v5",
    model_kwargs={"trust_remote_code": True, "device": "cuda"},
)


#calcolo metrica
def answer_similarity_score(filename):

    with open(filename, "r") as f: # Caricamento dei dati dal file JSON
        json_data = json.load(f)

    samples = create_samples_from_dataset(json_data["data"])
    dataset = EvaluationDataset(samples=samples)

    metrics = [answer_similarity]
    try:
        # Valuta il modello
        results = evaluate(
            llm=evaluator_llm,
            embeddings=hf,
            dataset=dataset,
            metrics=metrics,
        )

        #print(f"Answer similarity: {results}")
    except Exception as e:
        print(f"Errore durante la valutazione: {e}")
        results = None
    return results

# Caricamento dei dati
#dataset_zephyr_11_stella.json ok
#dataset_mistral_11.json ok 
#dataset_llama_11_stella.json ok
# dataset_gemma_11_stella.json ok
#dataset_Llama-11.json ok
#dataset_Mistral-11.json ok
# dataset_2024-10-07_20-17-04.json ok
#dataset_gemma_11_.json ok

def main():
    filename = sys.argv[1]
    print(f"Test file: {filename}")
    result = answer_similarity_score(filename)
    print(f"Answer similarity score totale: {result}")
    data = {"test": filename, "similarity_score": result}

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