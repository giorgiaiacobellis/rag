import utils
import data
import os
import json
import datetime

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import VLLM
from ragas import evaluate

from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

os.environ["OPENAI_API_KEY"] = ("sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="ragTestServer"


# Caricamento dei dati
filename = "dataset_2024-09-27_18-23-43.json"
with open(filename, "r") as f: # Caricamento dei dati dal file JSON
    json_data = json.load(f)

ds  = Dataset.from_dict(json_data["data"])

evaluator = VLLM(
            model=data.config["llm"]["model"],
            top_p=data.config["llm"]["top_p"],
            max_tokens=data.config["llm"]["max_tokens"],
            temperature=data.config["llm"]["temperature"],
            stream=data.config["llm"]["stream"]
        )

embedder = HuggingFaceEmbeddings(
        model_name=data.config["embedder"]["model"],
        model_kwargs=data.config["embedder"]["model_kwargs"],
    )
try:
    # Valuta il modello
    results = evaluate(
        llm=evaluator,
        dataset=ds,
        embeddings=embedder,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_recall,
            context_precision
        ],
    )
    print(results)
except Exception as e:
    print(f"Errore durante la valutazione: {e}")
    results = None

'''
# Salva i risultati in un file
save_results = {"model_info": data.config["llm"], "evaluation_results": results}

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file = f"results_{timestamp}.json"
with open(results_file, "w") as f:
    json.dump(save_results, f, indent=4)

print(f"Results saved to {results_file}")
'''