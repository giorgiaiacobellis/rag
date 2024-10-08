import json
import os
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

def carica_file(url, filename):
    if not os.path.exists(filename):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Genera un'eccezione per codici di stato HTTP non validi (ad esempio 404)

            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    f.write(chunk)

            print(f"File scaricato con successo come {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Errore durante il download del file: {e}")
    else:
        print(f"Il file {filename} esiste già.")



def costruzione_retriever(filename):
    print("sto caricando i dati!")
    # Caricamento dei dati
    with open(filename, "r") as f: # Caricamento dei dati dal file JSON
        json_data = json.load(f)

    # Conversione dei dati in documenti Langchain
    documents = [
        Document(page_content=item["text"], metadata={"source": item["page_name"]})
        for item in json_data
    ]

    print("split dei dati!")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50) 
    splits = text_splitter.split_documents(documents) 
    print("lensplits", len(splits))

    #Vector DB
    print("caricamento embedder")
    embedder = HuggingFaceEmbeddings(
        model_name="dunzhang/stella_en_400M_v5",
        model_kwargs= {"trust_remote_code": True, "device": "cuda"},
    )

    print("generazione vectorDB")
    vectordb = Chroma(collection_name="test_vectordb",
                      embedding_function=embedder,
                      persist_directory="./test_vectordb")
    vectordb.reset_collection()
    
    for i in range(0, len(splits), 4000):
        print("altri 4000")
        if (i+4000) > len(splits):
            vectordb.from_documents(documents=splits[i:], 
                            collection_name="test_vectordb",
                            embedding=embedder,
                            persist_directory="./test_vectordb")
        else:
            vectordb.from_documents(documents=splits[i:i+4000], 
                                collection_name="test_vectordb",
                                embedding=embedder,
                                persist_directory="./test_vectordb")
        
    retriever = vectordb.as_retriever(search_type = "mmr", search_kwargs={ "k":5, "fetch_k": 50, "lambda_mult": 0})
    return retriever, vectordb


#carica_file("https://evilscript.eu/upload/files/new_dati_wiki_piemonte.json", "wiki_piemonte.json")
#test retriever cambiando embedder
retriever, vectordb = costruzione_retriever("wiki_piemonte.json")

#query = "Quali sono i piatti tipici piemontesi che dovrei assolutamente provare?"


retriever = vectordb.as_retriever(search_type = "mmr", search_kwargs={ "k":5, "fetch_k": 50, "lambda_mult": 0})
docs = retriever.invoke("Quali sono i piatti tipici piemontesi che dovrei assolutamente provare?")
print(docs)

