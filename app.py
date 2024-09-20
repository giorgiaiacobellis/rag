#from ragas.embeddings import HuggingfaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
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

  llm = HuggingFaceEndpoint(
  repo_id="mistralai/Mistral-Nemo-Instruct-2407",
  task="text-generation",
  max_new_tokens=512,
  do_sample=False,
  repetition_penalty=1.03,
  )


  pipe = pipeline(
      model=AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta"),
      tokenizer=AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta"),
      return_full_text=True,  # langchain expects the full text
      task='text-generation',
      temperature=0.1,
      repetition_penalty=1.1  # without this output begins repeating
      max_new_tokens = 512
  )

  evaluator = HuggingFacePipeline(pipeline=pipe)

  #utils.chat_with_chatbot(data.config1, "session_1")
  utils.evaluate_model(data.config1, evaluator, embed_model, data.questions, data.answers)
  utils.evaluate_model(data.config1, llm, embed_model, data.questions, data.answers) 


if __name__ == "__main__":
    main()