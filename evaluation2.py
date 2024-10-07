from langchain_huggingface import HuggingFaceEmbeddings
from ragas import SingleTurnSample, EvaluationDataset
import json
import data

from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas import evaluate
from langchain_community.llms.vllm import VLLM


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

evaluator = VLLM(
            model=data.config3["llm"]["model"],
            top_p=data.config3["llm"]["top_p"],
            max_new_tokens=data.config3["llm"]["max_new_tokens"],
            temperature=data.config3["llm"]["temperature"],
            top_k=data.config3["llm"]["top_k"],
            trust_remote_code= True
        )

hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"trust_remote_code": True, "device": "cuda"},
)

# Caricamento dei dati
filename = "dataset_prova.json"
with open(filename, "r") as f: # Caricamento dei dati dal file JSON
    json_data = json.load(f)

samples = create_samples_from_dataset(json_data["data"])
dataset = EvaluationDataset(samples=samples)

# Stampa i sample creati
for sample in samples:
    print(sample)

metrics = [ResponseRelevancy(), Faithfulness()]
try:
    # Valuta il modello
    results = evaluate(
        llm=evaluator,
        embeddings=hf,
        dataset=dataset,
        metrics=metrics,
        show_progress=False,
    )
    print(results)
except Exception as e:
    print(f"Errore durante la valutazione: {e}")
    results = None
