import utils

domande_turistiche = [
    # Cibo
    "Quali sono i piatti tipici piemontesi che dovrei assolutamente provare?",
    "Potresti consigliarmi alcuni ristoranti a Torino dove posso gustare la cucina tradizionale piemontese?",
    "Dove posso trovare i migliori vini piemontesi?",
    "Ci sono sagre o festival gastronomici in Piemonte nei prossimi mesi?",
    "Quali sono i prodotti tipici piemontesi che posso portare a casa come souvenir?",

    # Itinerari
    "Vorrei fare un itinerario di 3 giorni in Piemonte, cosa mi consigli?",
    "Quali sono i borghi più belli da visitare in Piemonte?",
    "Potresti suggerirmi un itinerario enogastronomico in Piemonte?",
    "Qual è il periodo migliore dell'anno per visitare le Langhe?",
    "Vorrei fare un viaggio in treno in Piemonte, quali sono le tratte più panoramiche?",

    # Eventi
    "Quali sono gli eventi culturali più importanti in Piemonte nei prossimi mesi?",
    "Ci sono concerti o festival musicali in programma in Piemonte?",
    "Dove posso trovare informazioni sugli eventi sportivi in Piemonte?",
    "Quali sono le sagre e le feste tradizionali più caratteristiche del Piemonte?",
    "Ci sono eventi speciali per famiglie con bambini in Piemonte?",

    # Descrizione di un luogo
    "Cosa posso vedere e fare a Torino in un giorno?",
    "Quali sono le attrazioni principali del Lago Maggiore?",
    "Potresti descrivermi l'atmosfera delle Langhe?",
    "Cosa rende speciale la città di Alba?",
    "Quali sono le caratteristiche del paesaggio montano piemontese?",

    # Attrazioni in una località specifica
    "Cosa posso visitare ad Asti?",
    "Quali sono i musei più interessanti di Alessandria?",
    "Cosa posso fare a Stresa oltre a visitare le isole Borromee?",
    "Quali sono le attività all'aria aperta che posso fare a Cuneo e dintorni?",
    "Ci sono castelli o palazzi storici da visitare a Vercelli?",

    # Altre domande
    "Quali sono i migliori mercatini di Natale in Piemonte?",
    "Dove posso fare shopping di prodotti artigianali piemontesi?",
    "Ci sono terme o centri benessere in Piemonte?",
    "Quali sono le località sciistiche più famose in Piemonte?",
    "Come posso raggiungere il Piemonte dall'aeroporto di Milano Malpensa?"
]

questions = [ "Quali sono i piatti tipici piemontesi che dovrei assolutamente provare?",
              "Quali sono i borghi più belli da visitare in Piemonte?"
              ]

answers = ["Ecco alcuni piatti tipici piemontesi da non perdere: Antipasti: Vitello tonnato, Bagna càuda Primi: Agnolotti del plin, Tajarin al tartufo, Risotto al Barolo Secondi: Brasato al Barolo, Bollito misto, Finanziera Dolci: Bonet, Torta di nocciole, Baci di dama Non dimenticare di accompagnare il tutto con i vini piemontesi come Barolo, Barbaresco e Nebbiolo!",
            "Ecco alcuni dei borghi più belli del Piemonte: Langhe e Roero: Barolo, Grinzane Cavour, La Morra, Neive Laghi: Orta San Giulio, Cannobio Altri: Saluzzo"
            ]



#----------------------------CONFIGURATIONS------------------------------
#Config1: zephyr-7b-beta
config1 = {
    'llm': {
      'provider': 'huggingface',
      'config': {
        'model': 'HuggingFaceH4/zephyr-7b-beta',
        'top_p': 0.7,
        'max_tokens': 1000,
        'temperature': 0.5,
        'stream': True,
        'prompt': utils.get_modified_prompt("Sei un assistente turistico specializzato nel Piemonte. il tuo obiettivo è fornire informazioni accurate, utili e interessanti ai turisti che desiderano visitare il Piemonte. rispondi a domande su attrazioni turistiche, eventi, itinerari, cucina tipica, trasporti, alloggi e altre informazioni utili per i turisti. Sii preparato a rispondere a domande aperte, richieste di consigli e suggerimenti personalizzati in base agli interessi e alle esigenze dei turisti. Usa un tono amichevole, accogliente e professionale. sii entusiasta di condividere le bellezze e le peculiarità del Piemonte. Adatta il tuo stile di comunicazione al pubblico di riferimento che può includere famiglie, coppie, viaggiatori solitari, appassionati di enogastronomia, amanti della natura, ecc. Utilizza le informazioni estratte dai siti web dei comuni del Piemonte e altre fonti affidabili per fornire risposte accurate e aggiornate. Se non sei sicuro di una risposta, ammettilo onestamente e suggerisci alte fonti di informazione o modalità di contatti  per ottenere ulteriori dettaglio.",
                                      "HuggingFaceH4/zephyr-7b-beta"),
        'local': True

      }
    },
    'embedder': {
      'provider': 'huggingface',
      'config': {
        'model': 'sentence-transformers/all-mpnet-base-v2'
      }
    },
    'vectordb': {
      'provider': 'chroma',
      'config': {
        'collection_name': 'my-collection',
        'dir': 'db',
        'allow_reset': False
      }
    }
}



#Config2: mistral-7b-instruct 
config2 = {
  'app':{
      'config': {
          'name': 'Mistral-7B-Instruct-v0.2',
      }

  },
  'llm': {
    'provider': 'huggingface',
    'config': {
      'model': 'mistralai/Mistral-7B-Instruct-v0.2',
      'top_p': 0.7,
      'max_tokens': 1000,
      'temperature': 0.5,
      'stream': True,
      'prompt': utils.get_modified_prompt("Sei un chatbot turistico specializzato nel Piemonte. il tuo obiettivo è fornire informazioni accurate, utili e interessanti ai turisti che desiderano visitare il Piemonte. rispondi a domande su attrazioni turistiche, eventi, itinerari, cucina tipica, trasporti, alloggi e altre informazioni utili per i turisti. Sii preparato a rispondere a domande aperte, richieste di consigli e suggerimenti personalizzati in base agli interessi e alle esigenze dei turisti. Usa un tono amichevole, accogliente e professionale. sii entusiasta di condividere le bellezze e le peculiarità del Piemonte. Adatta il tuo stile di comunicazione al pubblico di riferimento che può includere famiglie, coppie, viaggiatori solitari, appassionati di enogastronomia, amanti della natura, ecc. Utilizza le informazioni estratte dai siti web dei comuni del Piemonte e altre fonti affidabili per fornire risposte accurate e aggiornate. Se non sei sicuro di una risposta, ammettilo onestamente e suggerisci alte fonti di informazione o modalità di contatti  per ottenere ulteriori dettaglio.",
                                      "mistralai/Mistral-7B-Instruct-v0.2")
        #'local': True
    }
  },
  'embedder': {
    'provider': 'huggingface',
    'config': {
      'model': 'sentence-transformers/all-mpnet-base-v2'
    }
  },
  'vectordb': {
    'provider': 'chroma',
    'config': {
      'collection_name': 'my-collection',
      'dir': 'db',
      'allow_reset': False
    }
  }
}
