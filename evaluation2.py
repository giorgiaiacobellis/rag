from langchain_huggingface import HuggingFaceEmbeddings
from ragas import SingleTurnSample, EvaluationDataset
import json
import data
import os

from ragas.metrics import LLMContextRecall, faithfulness, FactualCorrectness, SemanticSimilarity, answer_similarity
from ragas import evaluate
import vllm
from langchain_community.llms.vllm import VLLM
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

os.environ["OPENAI_API_KEY"] = ("sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="RagasTest"


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
            trust_remote_code= True,
            max_new_tokens = 128000
        )

evaluator_llm = LangchainLLMWrapper(evaluator)

hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"trust_remote_code": True, "device": "cuda"},
)
embd = LangchainEmbeddingsWrapper(hf)

# Caricamento dei dati
filename = "dataset_2024-10-07_12-09-43.json"
with open(filename, "r") as f: # Caricamento dei dati dal file JSON
    json_data = json.load(f)

samples = create_samples_from_dataset(json_data["data"])
dataset = EvaluationDataset(samples=samples)


metrics = [faithfulness,answer_similarity]
try:
    # Valuta il modello
    results = evaluate(
        llm=evaluator_llm,
        embeddings=embd,
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_retries=64)
    )

    print(results)
except Exception as e:
    print(f"Errore durante la valutazione: {e}")
    results = None
