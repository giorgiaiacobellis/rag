import json
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document



def costruzione_retriever():
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
    print("lensplits", len(splits))

    #Vector DB
    print("caricamento embedder")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_kwargs= {"trust_remote_code": True, "device": "cuda"},
    )


    print("generazione vectorDB")
    vectordb = Chroma(collection_name="test_vectordb",
                      embedding_function=embedder,
                      persist_directory="./test_vectordb")
    #vectordb.reset_collection()

    vectordb.from_documents(documents=splits, 
                            collection_name="test_vectordb",
                            embedding=embedder,
                            persist_directory="./test_vectordb")
        
    retriever = vectordb.as_retriever(search_type = "similarity", search_kwargs={ "k":5, "fetch_k": 50, "lambda_mult": 0})
    return retriever, vectordb

#test retriever cambiando embedder
retriever, vectordb = costruzione_retriever()

query = "Quali sono i piatti tipici piemontesi che dovrei assolutamente provare?"
#docs = vectordb.similarity_search(query)

docs = retriever.invoke("Quali sono i piatti tipici piemontesi che dovrei assolutamente provare?")
print(docs[3].page_content)