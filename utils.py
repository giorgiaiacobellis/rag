from embedchain import App
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
)

# Dimensione del batch per il caricamento dei dati
BATCH_SIZE = 100


# funzione che adatta il prompt in base al modello
def get_modified_prompt(system: str, model_name: str) -> str:
    if "gemma" in model_name.lower() and "-it" in model_name.lower():
        return f"<bos><start_of_turn>user\n{system}nContext information:\n----------------------\n$context\n----------------------\n$history\n $query<end_of_turn>\n<start_of_turn>model\n Answer: "

    elif "zephyr-7b-beta" in model_name.lower():
        return f"<|system|>{system}nContext information:\n----------------------\n$context\n----------------------\n$history\n</s><|user|>\n\nQuery: $query</s><|assistant|>\n Answer: "

    elif "meta" in model_name.lower():
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}nContext information:\n----------------------\n$context\n----------------------\n$history\n<|eot_id|><|start_header_id|>user<|end_header_id|>Query: $query<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n Answer: "

    elif "c4ai" in model_name.lower():
        return f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{system}nContext information:\n----------------------\n$context\n----------------------\n$history\n$query<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|> Answer: "

    elif "jamba" in model_name.lower():
        return f"{system} $query\n"

    return f"[INST]{system}nContext information:\n----------------------\n$context\n----------------------\n$history\n $query[/INST]\n Answer:"


# funzione per caricare i dati in batch nel vector db (da non usare)
def load_data_in_batches(filename, app):
    """Carica i dati dal file JSON in batch e li aggiunge a Embedchain."""

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(len(data))
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]

        # Formatta i dati per Embedchain
        formatted_batch = [
            {"text": item["text"]}
            for item in batch  # Estrai il campo "text"
        ]
        formatted_batch = json.dumps(formatted_batch)
        app.add(formatted_batch)  # Aggiungi il batch a Embedchain
        print(f"Aggiunto batch {i // BATCH_SIZE + 1} di {len(data) // BATCH_SIZE}")


# funzione che genera il dataset di [query, answer, context, ground_truth]
def generate_responses(config, test_questions, test_answers):
    answers = []
    contexts = []

    app = App.from_config(config=config) #nel caso rimuovere
    for q in test_questions:
        
        answer, context = app.query(q, citations=True)
        answers.append(answer[answer.find("Answer:") + len("Answer") :].strip())
        contexts.append([tupla[0] for tupla in context])

    dataset_dict = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
    }
    if test_answers is not None:
        dataset_dict["ground_truth"] = test_answers
    ds = Dataset.from_dict(dataset_dict)
    return ds


# funzione per testare il modello in una conversazione
def chat_with_chatbot(config, session):
    app = App.from_config(config=config)  # create app from config
    filename = "chatbot_responses.json"
    app.delete_all_chat_history(app.id)  # clean della chat history

    while True:
        try:
            input_query = input(
                "\nCosa vuoi chiedere al chatbot? (Digita 'esci' per uscire) "
            )

            # Controlla se l'utente vuole uscire
            if input_query.lower() == "esci":
                break

            answer = app.chat(
                input_query, session_id=session, citations=False
            )  # Assumiamo che 'app' sia definito correttamente
            reduced = answer[answer.find("Answer:") + len("Answer:") :].strip()
            print(f"{reduced}\n\n")

            # Crea un dizionario per memorizzare i dati dell'interazione corrente
            data_to_save = {
                "responses": [
                    {
                        "config_data": app.llm.config.serialize(),
                        "query": input_query,
                        "response": answer,
                    }
                ]
            }

            # Prova a caricare i dati esistenti dal file, se esiste
            try:
                with open(filename, "r") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = []  # Se il file non esiste, crea una lista vuota

            # Aggiungi i nuovi dati alla lista esistente
            existing_data.append(data_to_save)

            # Apri il file in modalità 'write' per sovrascrivere i dati
            with open(filename, "w") as f:
                json.dump(existing_data, f, indent=4)

            # print("Dati dell'interazione salvati correttamente in", filename)

        except EOFError:
            print("Fine dell'input. Arrivederci!")
            break


# funzione per valutare il modello
def evaluate_model(
    config, evaluator, test_questions, test_answers, filename="evaluation_results.json"
):
    app = App.from_config(config=config)  # create app from config
    llm_config = app.llm.config.serialize()

    app.delete_all_chat_history(app.id)  # clean della chat history

    ds = generate_responses(
        app, test_questions, test_answers
    )  # genera le risposte del modello

    try:
        # Valuta il modello
        results = evaluate(
            llm=evaluator,
            dataset=ds,
            metrics=[
                faithfulness,
                answer_relevancy,
                answer_correctness,
                # context_recall,
                # context_precision
            ],
        )
        print(results)
    except Exception as e:
        print(f"Errore durante la valutazione: {e}")
        results = None

    # Salva i risultati in un file
    save_results = {"model_info": llm_config, "evaluation_results": results}

    try:
        with open(filename, "r") as f:
            existing_result = json.load(f)
    except FileNotFoundError:
        existing_result = []  # Se il file non esiste, crea una lista vuota

    # Aggiungi i nuovi dati alla lista esistente
    existing_result.append(save_results)

    with open(filename, "w") as f:
        json.dump(existing_result, f, indent=4)

    print(f"Results saved to {filename}")

    return results
