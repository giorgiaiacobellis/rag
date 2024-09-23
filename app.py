from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
import utils
import data
import os
import json 


os.environ["OPENAI_API_KEY"] = (
    "sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A"
)
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = (
    "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"  # "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"
)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def main():
    '''
    pipe = pipeline(
        model=AutoModelForCausalLM.from_pretrained("arcee-ai/Llama-3.1-SuperNova-Lite"),
        tokenizer=AutoTokenizer.from_pretrained("arcee-ai/Llama-3.1-SuperNova-Lite"),
        return_full_text=True,  # langchain expects the full text
        task="text-generation",
        temperature=0.5,
        repetition_penalty=1.1,  # without this output begins repeating
        max_new_tokens=512,
        device=0,
    )

    evaluator1 = HuggingFacePipeline(pipeline=pipe)
    '''
    ds = utils.generate_responses(data.config, data.questions, data.answers)
    
    with open('answers.json', 'w') as json_file:
        json.dump(ds, json_file, indent=4)
    
    #utils.evaluate_model(data.config, evaluator1, data.questions, data.answers)


if __name__ == "__main__":
    main()
