from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import VLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
import json
import utils
import data

os.environ["OPENAI_API_KEY"] = ("sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")  #"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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="ragTestServer"

#VARIABILI DI CONTROLLO
SPLIT = 1  # vale zero se lo split è da eseguire altimenti 1
VECTORDB = 1 #vale 0 se il vectordb è da costruire altrimenti 1



def main():
    # Caricamento del modello LLM
    
    print("ehi fratello, sto caricando il modello!")
    llm = VLLM(
        model="HuggingFaceH4/zephyr-7b-beta",
        top_p=1,
        max_tokens=1000,
        temperature=0.5,
        stream=True
    )

    
    with open("cleaned_data.json", "r") as f: # Caricamento dei dati dal file JSON
        json_data = json.load(f)

    # Conversione dei dati in documenti Langchain
    documents = [
        Document(page_content=item["text"], metadata={"source": item["page_name"]})
        for item in json_data
    ]
    

    if SPLIT == 0:
        print("yo gang, sto caricando i dati!")
        # Caricamento dei dati

        print("e diamogliela na splittata a sti docssss")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

    #Vector DB
    print("lo facciamo o non lo facciamo sto embedder?")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"trust_remote_code": True, "device": 0},
    )

    print("grande fra, hai raggiunto il livello VectorDB, ecco a te il tuo calippo!")
    vectordb = Chroma(collection_name="turism_collection",
                      embedding_function=embedder,
                       persist_directory="./chroma_langchain_db")
    
    if VECTORDB == 0:
        vectordb.from_documents(documents=splits, 
                                collection_name="turism_collection",
                                embedding=embedder,
                                persist_directory="./chroma_langchain_db")
    
    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Sei un assistente turistico specializzato nel Piemonte. il tuo obiettivo è fornire informazioni accurate, utili e interessanti ai turisti che desiderano visitare il Piemonte. rispondi a domande su attrazioni turistiche, eventi, itinerari, cucina tipica, trasporti, alloggi e altre informazioni utili per i turisti. Sii preparato a rispondere a domande aperte, richieste di consigli e suggerimenti personalizzati in base agli interessi e alle esigenze dei turisti. Usa un tono amichevole, accogliente e professionale. sii entusiasta di condividere le bellezze e le peculiarità del Piemonte. Adatta il tuo stile di comunicazione al pubblico di riferimento che può includere famiglie, coppie, viaggiatori solitari, appassionati di enogastronomia, amanti della natura, ecc. Utilizza le informazioni estratte dai siti web dei comuni del Piemonte e altre fonti affidabili per fornire risposte accurate e aggiornate. Se non sei sicuro di una risposta, ammettilo onestamente e suggerisci alte fonti di informazione o modalità di contatti per ottenere ulteriori dettaglio" "{context}"),
            ("human", "{input}"),
        ]
    )

    print("chattiamo va! Boia faus!")

    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    '''
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    '''
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Esecuzione della RAG per ogni domanda
    #responses = rag_chain.batch(data.questions)

    # Stampa le risposte
    for question in data.questions:
        response = rag_chain.invoke({"input": question})
        print(f"Domanda: {question}")
        print(f"Risposta: {response}\n")
       

if __name__ == "__main__":
    main()
