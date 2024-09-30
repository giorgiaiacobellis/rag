import utils

questions = [
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
    "Come posso raggiungere il Piemonte dall'aeroporto di Milano Malpensa?",
]

ground_truth = [
    # Cibo
    "Antipasti: Vitello tonnato, battuta di fassona al coltello, acciughe al verde, peperoni con la bagna cauda. Primi: Agnolotti del plin, tajarin al tartufo, risotto al Barolo, panissa. Secondi: Brasato al Barolo, bollito misto alla piemontese, finanziera, polenta concia. Dolci: Bonet, torta di nocciole, baci di dama, panna cotta.",
    "Ristorante Del Cambio: Storico ristorante nel cuore di Torino, offre un'esperienza culinaria di alto livello con piatti della tradizione rivisitati in chiave moderna. Tre Galline: Locale caratteristico con atmosfera rustica, propone piatti tipici piemontesi preparati con ingredienti di qualità. La Taverna dei Mercanti: Situata in un antico palazzo, offre un ambiente elegante e una cucina piemontese raffinata. Cianci Piola Caffè: Trattoria storica con cucina casalinga e porzioni abbondanti, ideale per un pranzo informale.",
    "Langhe, Roero e Monferrato: Queste zone sono famose per la produzione di Barolo, Barbaresco, Nebbiolo, Barbera, Dolcetto e molti altri vini pregiati. Visita le cantine vinicole per degustazioni e acquisti. Enoteche specializzate: A Torino e nelle principali città troverai enoteche con una vasta selezione di vini piemontesi, dove potrai ricevere consigli e acquistare bottiglie.",
    "Fiera Internazionale del Tartufo Bianco d'Alba (Alba, ottobre-novembre): Evento imperdibile per gli amanti del tartufo, con degustazioni, mostre mercato e eventi culturali. Bagna Cauda Day (novembre): Giornata dedicata alla celebre salsa piemontese, con eventi e degustazioni in varie località. Sagra del Bollito (inverno): Diverse località organizzano sagre dedicate al bollito misto alla piemontese, un piatto ricco e tradizionale. Cioccolatò (Torino, novembre): Festival dedicato al cioccolato, con degustazioni, laboratori e eventi per golosi di tutte le età.",
    "Tartufo bianco d'Alba: Se visiti il Piemonte in autunno, non perdere l'occasione di acquistare questo pregiato prodotto. Nocciole del Piemonte IGP: Le nocciole piemontesi sono famose per la loro qualità e sono utilizzate in molti dolci e prodotti tipici. Vini piemontesi: Porta a casa una bottiglia di Barolo, Barbaresco o di un altro vino locale per ricordare il tuo viaggio. Formaggi tipici: Castelmagno, Bra, Raschera, Toma Piemontese sono solo alcuni dei formaggi da provare e portare a casa. Salumi: La salsiccia di Bra è un salume tipico da non perdere. Grissini: I grissini torinesi sono un classico della tradizione piemontese. Gianduiotti e altri prodotti a base di cioccolato: Torino è famosa per il cioccolato, quindi non dimenticare di acquistare qualche dolce souvenir.",

    # Itinerari
    "Giorno 1: Torino. Visita il centro storico con Piazza Castello, Palazzo Reale e il Duomo. Ammira la Mole Antonelliana e visita il Museo Egizio. Giorno 2: Langhe e Roero. Esplora i suggestivi borghi di Barolo, Barbaresco e Neive, immergiti nei vigneti e visita alcune cantine vinicole per degustazioni. Giorno 3: Lago Maggiore. Fai una crociera alle Isole Borromee e visita Stresa o Verbania, due incantevoli cittadine sul lago.",
    "Neive: Borgo medievale immerso nelle Langhe, famoso per i suoi vini e la sua architettura. Barolo: Patria dell'omonimo vino, offre panorami mozzafiato sui vigneti e un castello medievale. Barbaresco: Altro borgo celebre per il vino, con un'antica torre e viste panoramiche. Monforte d'Alba: Arroccato su una collina, offre scorci suggestivi e un'atmosfera romantica. Cherasco: Città fortificata con un centro storico ben conservato e una tradizione gastronomica rinomata. Orta San Giulio: Incantevole borgo sul Lago d'Orta, con un'isola pittoresca e un'atmosfera rilassata. Ricetto di Candelo: Unico nel suo genere, è un antico borgo fortificato interamente conservato.",
    "Langhe e Roero: Visita cantine vinicole, partecipa a degustazioni di Barolo e Barbaresco, e cena in ristoranti stellati che propongono piatti della tradizione rivisitati in chiave moderna. Monferrato: Esplora i vigneti del Monferrato, assaggia Barbera e Grignolino, visita castelli e partecipa a sagre gastronomiche. Torino e dintorni: Scopri i mercati alimentari, le cioccolaterie storiche, i caffè eleganti e i ristoranti tipici della città e dei dintorni.",
    "Autunno (settembre-novembre): La vendemmia, la Fiera del Tartufo Bianco d'Alba e i colori autunnali dei vigneti rendono questo il periodo ideale per visitare le Langhe. Primavera (aprile-giugno): La fioritura delle vigne, le temperature miti e la minore affluenza turistica rendono la primavera un'altra ottima stagione per visitare la regione.",
    "Ferrovia Vigezzina-Centovalli (Domodossola-Locarno): Attraversa paesaggi mozzafiato tra montagne, gole e cascate. Linea Torino-Ceres: Offre viste panoramiche sulla Val di Susa e sulle Alpi. Linea del Canavese (da Torino a Pont Canavese): Attraversa colline, vigneti e borghi caratteristici. Linea Asti-Acqui Terme: Percorre le dolci colline del Monferrato, offrendo scorci suggestivi.",

    # Eventi
    "Ti consiglio di consultare il sito ufficiale di Turismo Piemonte o i siti dei comuni per un calendario aggiornato degli eventi. Alcuni eventi ricorrenti di rilievo sono: Salone del Libro (Torino, maggio), MITO SettembreMusica (Torino e Milano, settembre), Luci d'Artista (Torino, inverno).", 
    "Per informazioni aggiornate su concerti e festival musicali, consulta i siti di ticketing online o i siti dei principali locali e teatri. Alcuni festival musicali ricorrenti sono: Collisioni (Barolo, luglio), Kappa FuturFestival (Torino, luglio), Movement Torino Music Festival (Torino, ottobre).",
    "Puoi trovare informazioni sugli eventi sportivi sui siti ufficiali delle squadre e delle federazioni sportive, nelle sezioni sportive dei principali quotidiani e siti di notizie locali, e su siti specializzati in eventi sportivi.",
    "Palio di Asti (Asti, settembre): Storica corsa di cavalli che si svolge nella suggestiva cornice di Piazza Alfieri. Carnevale di Ivrea (Ivrea, febbraio): Celebre per la sua battaglia delle arance, è uno dei carnevali più antichi e particolari d'Italia. Festa della Barbera (Nizza Monferrato, maggio): Evento dedicato al vino Barbera, con degustazioni, eventi culturali e folclore. Fiera del Bue Grasso (Carrù, dicembre): Tradizionale fiera dedicata al bue grasso di Carrù, con esposizione di animali, degustazioni e eventi folcloristici.",
    "Zoom Torino (parco zoologico), Museo Egizio (Torino) con laboratori didattici, Parco Avventura Veglio (Biella), Castello di Masino (Caravino) con eventi e attività per bambini.",

    # Descrizione di un luogo
    "Visita Piazza Castello, Palazzo Reale, Duomo di Torino, Mole Antonelliana. Passeggiata lungo il Po e nei parchi cittadini. Museo Egizio, Museo del Cinema, GAM (Galleria d'Arte Moderna). Pranzo o cena in un ristorante tipico piemontese.",
    "Isole Borromee (Isola Bella, Isola Madre, Isola dei Pescatori), Rocca di Angera, Giardino Botanico Alpinia, Villa Taranto, Stresa, Verbania, Arona.",
    "Colline ondulate ricoperte di vigneti, borghi medievali, castelli, cantine vinicole. Atmosfera rilassata e accogliente, legata alla tradizione enogastronomica e alla vita rurale. Paesaggi suggestivi, soprattutto in autunno con i colori del foliage.",
    "Capitale del Tartufo Bianco d'Alba, centro storico medievale ben conservato, numerose torri e chiese antiche, eventi enogastronomici di rilievo, atmosfera elegante e vivace.",
    "Alpi occidentali con cime imponenti (come il Monte Rosa e il Monviso), valli alpine con laghi, boschi e pascoli, parchi naturali (come il Parco Nazionale del Gran Paradiso e il Parco Naturale delle Alpi Marittime), stazioni sciistiche rinomate (come Sestriere, Bardonecchia, Sauze d'Oulx).", 

    # Attrazioni in una località specifica
    "Asti: Piazza Alfieri e centro storico, Cattedrale di Santa Maria Assunta, Battistero di San Pietro, Torre Troyana, Museo Paleontologico, Cantine vinicole.",
    "Musei ad Alessandria: Museo Civico, Museo del Cappello Borsalino, Museo Etnografico della Gambarina, Museo della Scienza e della Tecnica, Pinacoteca Civica.",
    "Stresa oltre le isole: Passeggiata sul lungolago, Giardino Botanico Alpinia, Villa Pallavicino, Funivia del Mottarone, escursioni nei dintorni.",
    "Attività all'aperto a Cuneo e dintorni: Escursioni e trekking nelle Alpi Marittime, mountain bike, rafting e kayak, arrampicata sportiva, sci in inverno.",
    "Castelli e palazzi a Vercelli: Castello Visconteo, Basilica di Sant'Andrea, Palazzo Centori, Torre dell'Angelo.",

    # Altre domande
    "Mercatini di Natale: Torino (Piazza Castello, Borgo Dora), Asti, Govone, Santa Maria Maggiore, Ornavasso.",
    "Shopping artigianale: Mercati locali, botteghe artigiane nei centri storici, negozi specializzati in prodotti tipici.",
    "Terme e centri benessere: Acqui Terme, Lurisia, Pré Saint Didier, Vinadio, Agliano Terme.",
    "Località sciistiche: Sestriere, Bardonecchia, Sauze d'Oulx, Limone Piemonte, Alagna Valsesia, Macugnaga.",
    "Raggiungere il Piemonte da Malpensa: Treno Malpensa Express per Milano Centrale, poi treno per Torino o altre città piemontesi. Autobus diretti per Torino. Taxi o transfer privato."
    ]

# ----------------------------CONFIGURATIONS------------------------------
# Config1: zephyr-7b-beta

config = {
    "llm": {
            "model": "HuggingFaceH4/zephyr-7b-beta",
            "top_p": 1,
            "max_tokens": 1000,
            "temperature": 0.5,
            "stream": True,
            "prompt": utils.get_modified_prompt(
                "Sei un assistente turistico specializzato nel Piemonte. il tuo obiettivo è fornire informazioni accurate, utili e interessanti ai turisti che desiderano visitare il Piemonte. rispondi a domande su attrazioni turistiche, eventi, itinerari, cucina tipica, trasporti, alloggi e altre informazioni utili per i turisti. Sii preparato a rispondere a domande aperte, richieste di consigli e suggerimenti personalizzati in base agli interessi e alle esigenze dei turisti. Usa un tono amichevole, accogliente e professionale. sii entusiasta di condividere le bellezze e le peculiarità del Piemonte. Adatta il tuo stile di comunicazione al pubblico di riferimento che può includere famiglie, coppie, viaggiatori solitari, appassionati di enogastronomia, amanti della natura, ecc. Utilizza le informazioni estratte dai siti web dei comuni del Piemonte e altre fonti affidabili per fornire risposte accurate e aggiornate. Se non sei sicuro di una risposta, ammettilo onestamente e suggerisci alte fonti di informazione o modalità di contatti  per ottenere ulteriori dettaglio.",
                "HuggingFaceH4/zephyr-7b-beta"
            ),
    },
    "embedder": {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "model_kwargs": {"trust_remote_code": True, "device": 0},
    },

    "vectordb": {
            "collection_name": "turism_collection",
            "persist_directory":"./chroma_langchain_db",
            "allow_reset": False,
    },
}
