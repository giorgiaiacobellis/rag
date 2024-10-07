from langchain_huggingface import HuggingFaceEmbeddings
from ragas import SingleTurnSample, EvaluationDataset
import json
import data
import os

from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity, answer_similarity
from ragas import evaluate
from langchain_community.llms.vllm import VLLM
from ragas.llms import LangchainLLMWrapper

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
            max_new_tokens=4000,
            trust_remote_code= True
        )

evaluator_llm = LangchainLLMWrapper(evaluator)

hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"trust_remote_code": True, "device": "cuda"},
)

# Caricamento dei dati
filename = "dataset_2024-10-06_19-21-42.json"
with open(filename, "r") as f: # Caricamento dei dati dal file JSON
    json_data = json.load(f)

samples = create_samples_from_dataset(json_data["data"])
dataset = EvaluationDataset(samples=samples)


metrics = [Faithfulness(),answer_similarity]
try:
    # Valuta il modello
    results = evaluate(
        llm=evaluator,
        embeddings=hf,
        dataset=dataset,
        metrics=metrics,
    )

    print(results)
except Exception as e:
    print(f"Errore durante la valutazione: {e}")
    results = None
