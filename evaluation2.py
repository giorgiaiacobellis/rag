from langchain_huggingface import HuggingFaceEmbeddings
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
import json
import os


from ragas.metrics import answer_similarity
from ragas import evaluate
from langchain_community.llms.vllm import VLLM
from ragas.llms import LangchainLLMWrapper


os.environ["OPENAI_API_KEY"] = ("sk-proj-aVtM"+"47AoCct6IkxYRd7kUdwAQSCO8DmcZ2Ht88YqVKmOsRUFGk6fT0aVQeAdT4M9j8aDv8pFiUT3BlbkFJ4EjTuuQ"+"_MbmTWCik4mPzQ9f8YKQcMWlcBu1boiwsnLSqPSDB2HGjuwbUDbUPx6lBFK85uYElkA")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ("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")
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
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            trust_remote_code= True,
            max_new_tokens=2000,
            #n=1
        )

evaluator_llm = LangchainLLMWrapper(evaluator)

hf = HuggingFaceEmbeddings(
    model_name="dunzhang/stella_en_400M_v5",
    model_kwargs={"trust_remote_code": True, "device": "cuda"},
)

# Caricamento dei dati
#dataset_zephyr_11_stella.json
#dataset_mistral_11.json
#dataset_llama_11_stella.json
# dataset_gemma_11_stella.json
#dataset_Llama-11.json
#dataset_Mistral-11.json
#
filename = "dataset_zephyr_11_stella.json"
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

    print(results)
except Exception as e:
    print(f"Errore durante la valutazione: {e}")
    results = None