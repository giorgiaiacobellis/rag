#from ragas.embeddings import HuggingfaceEmbeddings
#from ragas.llms import LangchainLLMWrapper
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
import utils
import data
import os



os.environ['OPENAI_API_KEY'] = 'sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A'
os.environ["HUGGINGFACE_ACCESS_TOKEN"] =  "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" #"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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

#app.reset()
# Carica i dati in batch da cleaned_data.json
#load_data_in_batches("cleaned_data.json", app)
#print("Caricamento dati completato!")


def main():
  
  # embedding model
  embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

  pipe = pipeline(
      model=AutoModelForCausalLM.from_pretrained("arcee-ai/Llama-3.1-SuperNova-Lite"),
      tokenizer=AutoTokenizer.from_pretrained("arcee-ai/Llama-3.1-SuperNova-Lite"),
      return_full_text=True,  # langchain expects the full text
      task='text-generation',
      temperature=0.5,
      repetition_penalty=1.1,  # without this output begins repeating
      max_new_tokens = 512
  )

  pipe2 = pipeline(
      model=AutoModelForCausalLM.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407"),
      tokenizer=AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407"),
      return_full_text=True,  # langchain expects the full text
      task='text-generation',
      #temperature=0.5,
      repetition_penalty=1.03,  # without this output begins repeating
      max_new_tokens = 512
  )

  evaluator1 = HuggingFacePipeline(pipeline=pipe)
  evaluator2 = HuggingFacePipeline(pipeline=pipe2)

  #utils.chat_with_chatbot(data.config1, "session_1")
  utils.evaluate_model(data.config1, evaluator1, embed_model, data.questions, data.answers)
  utils.evaluate_model(data.config1, evaluator2, embed_model, data.questions, data.answers) 


if __name__ == "__main__":
    main()