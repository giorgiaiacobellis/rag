import os
import json
import datetime

from datasets import Dataset
from ragas import evaluate
from langchain_community.llms.vllm import VLLM
from ragas.llms import LangchainLLMWrapper
from ragas.llms.prompt import Prompt
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
filename = "dataset_prova.json"
with open(filename, "r") as f: # Caricamento dei dati dal file JSON
    json_data = json.load(f)

ds  = Dataset.from_dict(json_data["data"])
#ds.remove_columns(["ground_truth"])

new_prompt = Prompt

evaluator =  VLLM(
    model="TheBloke/LLaMA2-13B-Tiefighter-AWQ",
    trust_remote_code=True,
    temperature=0.1,
    vllm_kwargs={"quantization": "awq"},
)


#faithfulness.nli_statements_message = nli_statement_message_new
#answer_correctness.max_retries=3
#answer_correctness.llm = evaluator
#faithfulness.long_form_answer_prompt = long_form_answer_prompt_new

try:
    # Valuta il modello
    results = evaluate(
        llm=evaluator,
        dataset=ds,
        metrics=[
            answer_correctness
        ],
        show_progress=False,
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


'''
Lista modelli testati come evaluators: 
- ybelkada/Mixtral-8x7B-Instruct-v0.1-AWQ
- TheBloke/Llama-2-13B-chat-GGM
- TheBloke/Llama-2-7b-Chat-AWQ
- hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
- meta-llama/Llama-2-13b-hf
- TheBloke/LLaMA2-13B-Tiefighter-AWQ"
'''