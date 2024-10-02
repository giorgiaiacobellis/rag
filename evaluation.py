import utils
import data
import os
import json
import datetime

from datasets import Dataset
from ragas import evaluate
from langchain_community.llms import VLLM
from ragas.metrics import Faithfulness
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
#ds.remove_columns(["contexts"])



long_form_answer_prompt_new = Prompt(
    name="long_form_answer_new_v1",
    instruction='''[INST] <<SYS>> Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON.
                    The output should be a well-formatted JSON instance that conforms to the JSON schema below.
                    As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
                    the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
                    Here is the output JSON schema:
                    ```
                    {"type": "array", "items": {"$ref": "#/definitions/Statements"}, "definitions": {"Statements": {"title": "Statements", "type": "object", "properties": {"sentence_index": {"title": "Sentence Index", "description": "Index of the sentence from the statement list", "type": "integer"}, "simpler_statements": {"title": "Simpler Statements", "description": "the simpler statements", "type": "array", "items": {"type": "string"}}}, "required": ["sentence_index", "simpler_statements"]}}}
                    ```
                    Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).<</SYS>>\n'''"Question: {question}\n""Answer: {answer}\n""Sentences:{sentences}\n""[/INST]",
    
    input_keys=["question", "answer", "sentences"],
    output_key="analysis",
    language="italian",
    output_type="json",
)

nli_statement_message_new = Prompt(
    name="nli_statements_new_v1",
    instruction=" [INST] <<SYS>>Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.<</SYS>>\n""Context: {context}\n""Statement: {statements}\n""[/INST]",
    input_keys=["context", "statements"],
    output_key="answer",
    output_type="json",
    language="italian",
)

evaluator =  VLLM(
    model="TheBloke/Llama-2-7b-Chat-AWQ",
    trust_remote_code=True,
    max_new_tokens=2000,
    vllm_kwargs={"quantization": "awq"},
)

#faithfulness.nli_statements_message = nli_statement_message_new
faithfulness.max_retries=3
#faithfulness.long_form_answer_prompt = long_form_answer_prompt_new

try:
    # Valuta il modello
    results = evaluate(
        llm=evaluator,
        dataset=ds.select(range(31)),
        metrics=[
            faithfulness,
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