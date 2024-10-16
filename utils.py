import json
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.vllm import VLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document



def split_data(split_value): 
    if split_value == 0:
        print("sto caricando i dati!")
        # Caricamento dei dati
        with open("cleaned_data.json", "r") as f: # Caricamento dei dati dal file JSON
            json_data = json.load(f)

        # Conversione dei dati in documenti Langchain
        documents = [
            Document(page_content=item["text"], metadata={"source": item["page_name"]})
            for item in json_data
        ]

        print("split dei dati!")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
        splits = text_splitter.split_documents(documents) 
        return splits
    else:
        return []
    

def create_vector_db(vectordb_value, config, splits):
    #Vector DB
    print("caricamento embedder")
    embedder = HuggingFaceEmbeddings(
        model_name=config["embedder"]["model"],
        model_kwargs=config["embedder"]["model_kwargs"],
    )
    
    print("generazione vectorDB")
    vectordb = Chroma(collection_name=config["vectordb"]["collection_name"],
                      embedding_function=embedder,
                       persist_directory=config["vectordb"]["persist_directory"])
    
    if vectordb_value == 0: #da eseguire se si devono caricare i dati sul vectorDB
        vectordb.from_documents(documents=splits, 
                                collection_name=config["vectordb"]["collection_name"],
                                embedding=embedder,
                                persist_directory=config["vectordb"]["persist_directory"])
    
    retriever = vectordb.as_retriever(search_type = "mmr", search_kwargs= { "k":5, "fetch_k": 50, "lambda_mult": 0})
    return retriever


# funzione che adatta il prompt in base al modello
def get_modified_prompt(system: str, model_name: str) -> str:
    if "gemma" in model_name.lower() and "-it" in model_name.lower():
        return f"<bos><start_of_turn>user\n{system}\n""Context: {context}""Question: {input}""<end_of_turn>\n<start_of_turn>model\n"

    elif "zephyr-7b-beta" in model_name.lower():
        return f"<|system|>{system}<\s>\n<|user|>""Context: {context}\n Question: {input}<\s>\n""<|assistant|>\n"

    elif "meta" in model_name.lower():
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n""Context {context}\n Question: {input}<|eot_id|>""<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    elif "mistral" in model_name.lower():
         return f"<s>[INST]{system}[/INST]<\s>\n" "Question:{input}\n Context:{context}" "Answer: [/INST]"
    
    return f"[INST]<<SYS>>{system} <</SYS>> Question:""{input}""Context:""{context}" "Answer: [/INST]"


# funzione che genera il dataset di [query, answer, context, ground_truth]
def generate_chat(config):
    print("sto caricando il modello!")
    llm = VLLM(
            model=config["llm"]["model"],
            top_p=config["llm"]["top_p"],
            max_new_tokens=config["llm"]["max_new_tokens"],
            temperature=config["llm"]["temperature"],
            top_k=config["llm"]["top_k"],
            trust_remote_code= True
        )
    
    prompt = PromptTemplate.from_template(config["llm"]["prompt"])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    return question_answer_chain
