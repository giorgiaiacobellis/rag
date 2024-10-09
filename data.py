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
    "Ah, la cucina piemontese! Preparatevi a un'esperienza culinaria indimenticabile. Ecco alcuni piatti che non potete assolutamente perdere: Agnolotti del plin: piccoli ravioli ripieni di carne arrosto, brasato e verdure, conditi con burro e salvia o sugo d'arrosto. Una vera delizia!; Bagna cauda: un intingolo caldo a base di aglio, olio, acciughe e burro, in cui si immergono verdure crude di stagione. Un piatto conviviale e saporito!; Vitello tonnato: fette di vitello lessato ricoperte da una salsa a base di tonno, acciughe e capperi. Un classico intramontabile!; Brasato al Barolo: manzo brasato a lungo nel Barolo, un vino rosso corposo. Un piatto ricco e saporito, perfetto per le giornate fredde.; Fritto misto alla piemontese: un trionfo di fritti, dolce e salato insieme: carne, frattaglie, verdure, frutta e semolino dolce. Una vera bomba di gusto!; Bonet: un dolce al cucchiaio a base di cacao, amaretti e rum. Un finale perfetto per un pasto piemontese.",
    "Certo! A Torino avrete l'imbarazzo della scelta. Ecco alcuni suggerimenti: Trattoria della Posta: un locale storico con atmosfera tradizionale e piatti autentici. Osteria Antiche Sere: un'osteria accogliente con un'ottima selezione di vini. Ristorante del Cambio: un ristorante elegante con una cucina raffinata e una vista mozzafiato su Piazza Carignano. Caffè Mulassano: un locale storico famoso per i suoi tramezzini e la cucina tipica piemontese. Per un'esperienza più informale, vi consiglio di visitare il Mercato di Porta Palazzo, dove potrete gustare street food piemontese e acquistare prodotti locali.",
    "Il Piemonte è una terra di grandi vini! Per degustare i migliori vini piemontesi, vi consiglio di visitare le Langhe, il Roero e il Monferrato. Qui potrete visitare cantine storiche, partecipare a degustazioni e acquistare bottiglie direttamente dai produttori. Alcuni dei vini più famosi del Piemonte sono: Barolo: un vino rosso corposo e strutturato, perfetto per accompagnare carni rosse e formaggi stagionati. Barbaresco: un vino rosso elegante e raffinato, ideale per piatti a base di tartufo. Nebbiolo: un vino rosso fruttato e speziato, ottimo con salumi e formaggi. Dolcetto: un vino rosso giovane e fresco, perfetto per accompagnare piatti leggeri. Moscato d'Asti: un vino bianco dolce e aromatico, ideale per dessert e aperitivi.",    
    "Assolutamente sì! Il Piemonte è ricco di sagre e festival gastronomici durante tutto l'anno. Per conoscere gli eventi in programma nei prossimi mesi, vi consiglio di consultare il sito web della Regione Piemonte o dei singoli comuni che vi interessano. Tra gli eventi più famosi, vi segnalo: Fiera Internazionale del Tartufo Bianco d'Alba: ad Alba, in autunno. Sagra del Bollito di Carrù: a Carrù, in autunno. Bagna Cauda Day: in diverse località del Piemonte, in autunno.",
    "Ecco alcune idee per souvenir gastronomici piemontesi: Tartufo bianco d'Alba: il re dei tartufi, un prodotto pregiato e dal sapore inconfondibile. Nocciole del Piemonte IGP: nocciole tonde e gentili, perfette per dolci e snack. Grissini Rubatà: grissini stirati a mano, croccanti e fragranti. Cioccolato di Torino: un'eccellenza italiana, famoso in tutto il mondo. Vini piemontesi: Barolo, Barbaresco, Nebbiolo, Dolcetto, Moscato d'Asti... la scelta è vastissima!",
    
    # Itinerari
    "Sono felice di aiutarvi a organizzare il vostro itinerario di 3 giorni. Per darvi i consigli migliori, avrei bisogno di qualche informazione in più sui vostri interessi: preferite la natura, l'arte, la storia, il relax, l'enogastronomia? In ogni caso, ecco un itinerario di 3 giorni che tocca alcune delle principali attrazioni del Piemonte, che potrete poi personalizzare in base alle vostre preferenze: Giorno 1: Torino Mattina: Visita del centro storico di Torino, con Piazza Castello, Palazzo Reale, il Duomo e la Mole Antonelliana. Pomeriggio: Museo Egizio, uno dei più importanti al mondo. Sera: Cena in un ristorante tipico piemontese (ricordate i miei consigli precedenti!). Giorno 2: Langhe e Roero Mattina: Visita di Alba, capitale del tartufo bianco, con il suo centro storico medievale. Pomeriggio: Degustazione di vini in una cantina del Barolo o del Barbaresco. Sera: Cena in un agriturismo con vista sulle colline. Giorno 3: Laghi o montagne Opzione Laghi: Gita in battello sul Lago Maggiore o sul Lago d'Orta, visita di Stresa o Orta San Giulio. Opzione Montagne: Escursione nel Parco Nazionale del Gran Paradiso o nella Val di Susa.",
    "Il Piemonte è ricco di borghi incantevoli, ecco alcuni dei più belli: Barolo: circondato da vigneti, famoso per il suo vino omonimo. Neive: con le sue case medievali in pietra e le torri. Monforte d'Alba: arroccato su una collina, offre una vista panoramica sulle Langhe. Orta San Giulio: affacciato sul Lago d'Orta, con l'isola di San Giulio. Vogogna: nella Val d'Ossola, con il suo castello medievale.",   
    "Certo! Un itinerario enogastronomico in Piemonte potrebbe includere: Langhe e Roero: degustazione di Barolo, Barbaresco e Roero Arneis, visita di cantine e aziende agricole, pranzo in un'osteria tipica. Monferrato: degustazione di Barbera, Grignolino e Freisa, visita di castelli e borghi medievali, cena con piatti a base di tartufo bianco. Torino: visita del Mercato di Porta Palazzo, degustazione di cioccolato e gianduiotti, aperitivo con tramezzini e bicerin.",
    "Le Langhe sono belle in ogni stagione, ma il periodo migliore per visitarle è l'autunno, durante la vendemmia e la Fiera Internazionale del Tartufo Bianco d'Alba.",
    "Ecco alcune tratte ferroviarie panoramiche in Piemonte: Linea Torino-Cuneo: attraversa le Langhe e offre viste spettacolari sulle colline e i vigneti. Ferrovia Vigezzina-Centovalli: collega Domodossola a Locarno (Svizzera), attraversando la Val Vigezzo e le Centovalli. Linea del Canavese: da Torino a Pont Canavese, attraversa il Canavese e offre viste sulle montagne.",
   
    # Eventi
    "Il Piemonte è una regione ricca di eventi culturali durante tutto l'anno. Ecco alcuni degli eventi più importanti in programma nei prossimi mesi: VIEW CONFERENCE 2024 a Torino (14-19 ottobre), 'Dietro la Maschera' a Casale Monferrato (12 novembre - 3 dicembre), 'Pietro Francesco Guala (1698-1757)' a Camino (7 dicembre - 1 dicembre 2024), 'Criminis. Vittime e carnefici' a Casale Monferrato (9 settembre - 28 febbraio 2025). Per rimanere aggiornati su tutti gli eventi culturali in Piemonte, vi consiglio di consultare i siti web: in Piemonte in Torino: https://www.inpiemonteintorino.it/ Visit Piemonte: https://www.visitpiemonte.com/eventi",
    "Sì, certo! Il Piemonte offre un'ampia scelta di concerti e festival musicali durante tutto l'anno. Ecco alcuni eventi in programma: XV Festival Internazionale Alessandria Barocca e non solo… ad Alessandria, fino al 5 ottobre. Un festival che propone concerti di musica barocca e non solo, con artisti di fama internazionale. Vignale in Danza a Vignale Monferrato, fino al 3 novembre. Un festival dedicato alla danza contemporanea, con spettacoli, workshop e incontri. NITTO ATP FINALS 2024 a Torino, dal 10 al 17 novembre. Oltre al grande tennis, le Nitto ATP Finals offrono anche un ricco programma di eventi musicali e di intrattenimento. Vi consiglio di consultare i siti web dei comuni e delle province piemontesi per scoprire i concerti e i festival musicali in programma nella zona che vi interessa.",
    "Per informazioni sugli eventi sportivi in Piemonte, potete consultare i seguenti siti web: Visit Piemonte: https://www.visitpiemonte.com/eventi (selezionate la categoria 'Sport e Natura') I siti web dei comuni e delle province piemontesi: molti comuni pubblicano i calendari degli eventi sportivi sui loro siti web.",
    "Il Piemonte è una regione ricca di tradizioni, con numerose sagre e feste che si svolgono durante tutto l'anno. Ecco alcune delle più caratteristiche: Fiera Internazionale del Tartufo Bianco d'Alba ad Alba, in autunno. Un evento imperdibile per gli amanti del tartufo, con mercati, degustazioni e eventi culturali. Sagra del Bollito di Carrù a Carrù, in autunno. Una festa dedicata al bollito misto alla piemontese, un piatto tipico della tradizione culinaria locale. Bagna Cauda Day in diverse località del Piemonte, in autunno. Una giornata dedicata alla bagna cauda, un intingolo a base di aglio, olio e acciughe in cui si immergono verdure crude. Palio di Asti ad Asti, a settembre. Una corsa di cavalli che si svolge nella piazza principale della città, con un corteo storico in costumi medievali.",
    "Certo! Il Piemonte offre numerose attività e attrazioni per famiglie con bambini. Ecco alcuni esempi: 'Il Fantasma della Villa' a Torino Per tutto il 2024, il Museo di Arti Decorative Accorsi - Ometto si trasforma in un grande spazio ludico, con un percorso interattivo alla scoperta del museo e dei suoi 'fantasmi'. Baskin Day a Bielmonte Il 6 ottobre 2024. Una giornata dedicata al baskin, uno sport inclusivo che permette a persone con e senza disabilità di giocare insieme a basket. 'All'Oasi Zegna, tra estate e autunno' ad Oasi Zegna Fino al 30 ottobre. Un ricco programma di attività all'aria aperta per tutta la famiglia, come escursioni, laboratori e visite guidate. 'Wine & ebike Adventure' nelle Langhe Per tutto il 2024. Un tour guidato in e-bike per famiglie alla scoperta delle Langhe, tra vigneti, borghi e panorami mozzafiato. Per ulteriori informazioni su eventi per famiglie con bambini in Piemonte, vi consiglio di consultare i siti web: Kid Pass: https://kidpass.it/piemonte-turismo-famiglia-bambini/ Visit Piemonte: https://www.visitpiemonte.com/",

    # Descrizione di un luogo
    "Torino è una città ricca di storia, arte e cultura. Ecco un itinerario per sfruttare al meglio una giornata a Torino: Mattina: Piazza Castello: Iniziate la vostra visita da Piazza Castello, il cuore della città. Ammirate Palazzo Reale, Palazzo Madama e il Teatro Regio. Mole Antonelliana: Salite sulla Mole Antonelliana, simbolo di Torino, per godere di una vista panoramica mozzafiato sulla città e sulle Alpi. Museo Egizio: Immergetevi nell'antico Egitto al Museo Egizio, uno dei più importanti al mondo. Pomeriggio: Pranzo: Gustate un pranzo tipico piemontese in uno dei tanti ristoranti del Quadrilatero Romano. Passeggiata lungo il Po: Passeggiate lungo le rive del Po, ammirando i ponti e i palazzi storici. Museo Nazionale del Cinema: Se siete appassionati di cinema, non perdete il Museo Nazionale del Cinema, ospitato all'interno della Mole Antonelliana. Sera: Aperitivo: Godetevi un aperitivo torinese in uno dei tanti locali del centro storico. Cena: Concludete la giornata con una cena in un ristorante tipico piemontese.",
    "Il Lago Maggiore offre una varietà di attrazioni per tutti i gusti: Isole Borromee: Visitate le incantevoli Isole Borromee: Isola Bella, con il suo Palazzo Borromeo e i giardini all'italiana; Isola Madre, con il suo giardino botanico; e Isola dei Pescatori, con il suo pittoresco villaggio di pescatori. Stresa: Passeggiate lungo il lungolago di Stresa, ammirando la vista sulle isole e sulle montagne. Rocca di Angera: Visitate la Rocca di Angera, un'imponente fortezza medievale con un museo dedicato alla storia del Lago Maggiore. Giardini Botanici di Villa Taranto: Ammirate la bellezza dei Giardini Botanici di Villa Taranto, con una vasta collezione di piante provenienti da tutto il mondo. Attività sportive: Il Lago Maggiore offre numerose opportunità per praticare sport acquatici, come vela, windsurf e canottaggio.",
    "Le Langhe sono un territorio magico, caratterizzato da dolci colline ricoperte di vigneti, borghi medievali e castelli. L'atmosfera è rilassante e bucolica, ideale per chi cerca una vacanza all'insegna del relax e del buon cibo. In autunno, durante la stagione del tartufo bianco, le Langhe si animano di eventi e mercati.",
    "Alba è la capitale delle Langhe, famosa in tutto il mondo per il suo pregiato tartufo bianco. La città ha un affascinante centro storico medievale, con torri, palazzi e chiese. Alba è anche un importante centro enogastronomico, con numerosi ristoranti, osterie e cantine dove degustare i prodotti tipici della zona.",
    "Il Piemonte offre una grande varietà di paesaggi montani, dalle Alpi Marittime alle Alpi Graie, con vette imponenti, valli verdi e laghi alpini. Il Parco Nazionale del Gran Paradiso è un vero paradiso per gli amanti della natura, con la possibilità di avvistare animali selvatici come stambecchi, camosci e aquile. La Val di Susa offre paesaggi mozzafiato e numerose opportunità per praticare sport invernali ed estivi.",    
   
    # Attrazioni in una località specifica
    "Asti è una città ricca di storia e cultura, con un bel centro storico medievale. Ecco alcuni luoghi che vi consiglio di visitare: Piazza Alfieri: La piazza principale di Asti, dedicata al poeta Vittorio Alfieri. Qui si affacciano importanti edifici storici, come Palazzo Ottolenghi e il Teatro Alfieri. Cattedrale di Santa Maria Assunta: Un magnifico esempio di architettura gotica piemontese, con affreschi e sculture di pregio. Torre Troyana: Una delle torri medievali più alte d'Italia, offre una vista panoramica sulla città e sui dintorni. Cripta di Sant'Anastasio: Una cripta romanica con affreschi del XII secolo. Museo Paleontologico: Un museo che espone fossili di animali preistorici ritrovati nella zona di Asti. Palazzo Mazzetti: Un palazzo barocco che ospita il Museo Civico, con collezioni di arte, archeologia e storia locale.",
    "Alessandria offre una varietà di musei interessanti: Museo Civico: Un museo che racconta la storia della città, con reperti archeologici, opere d'arte e documenti storici. Museo Etnografico 'C'era una volta': Un museo dedicato alla cultura popolare del Piemonte, con oggetti e testimonianze della vita quotidiana di un tempo. Museo del Cappello Borsalino: Un museo dedicato al famoso cappello Borsalino, prodotto ad Alessandria fin dal 1857. Galleria d'Arte Moderna: Una galleria che espone opere di artisti italiani del XIX e XX secolo. Museo di Scienze Naturali: Un museo con collezioni di zoologia, botanica e mineralogia.",
    "Oltre alle bellissime Isole Borromee, Stresa offre altre interessanti attrazioni: Passeggiata sul lungolago: Godetevi una rilassante passeggiata sul lungolago, ammirando la vista sulle isole e sulle montagne. Funivia Stresa-Alpino-Mottarone: Salite in funivia sul Mottarone per godere di un panorama mozzafiato sul Lago Maggiore e sulle Alpi. Giardino Botanico Alpinia: Visitate il Giardino Botanico Alpinia, con una ricca collezione di piante alpine. Eremo di Santa Caterina del Sasso: Raggiungete l'Eremo di Santa Caterina del Sasso, un antico monastero costruito sulla roccia a picco sul lago. Shopping e relax: Stresa offre una varietà di negozi, bar e ristoranti dove trascorrere il tempo libero.",
    "Cuneo e dintorni offrono numerose opportunità per gli amanti della natura e dello sport: Escursionismo: Percorrete i sentieri del Parco Naturale Alpi Marittime o della Valle Gesso, ammirando paesaggi alpini mozzafiato. Ciclismo: Pedalate lungo le strade panoramiche delle Langhe e del Roero, o cimentatevi in percorsi più impegnativi sulle montagne. Arrampicata: La Valle Gesso offre diverse falesie per l\'arrampicata sportiva. Sci: In inverno, le stazioni sciistiche di Limone Piemonte e Prato Nevoso offrono piste per tutti i livelli. Rafting e kayak: Provate l\'emozione del rafting o del kayak sul fiume Stura di Demonte.",
    "Certo! Vercelli ha un ricco patrimonio storico e architettonico. Ecco alcuni castelli e palazzi che vi consiglio di visitare: Castello Visconteo-Sforzesco: Un imponente castello medievale, oggi sede del Museo Borgogna. Basilica di Sant'Andrea: Una basilica romanica con un magnifico campanile. Palazzo Pasta: Un elegante palazzo neoclassico, oggi sede della Pinacoteca Civica. Abbazia di Sant'Agnese: Un'antica abbazia benedettina, con un chiostro e una chiesa romanica.",

    # Altre domande
    "Il Piemonte si trasforma in un luogo magico durante il periodo natalizio, con tanti mercatini che offrono un'atmosfera festosa e prodotti artigianali unici. Ecco alcuni dei migliori mercatini di Natale in Piemonte: Mercatini di Natale di Govone: Nel magnifico scenario del Castello Reale di Govone, questo mercatino offre un'ampia selezione di prodotti artigianali, decorazioni natalizie e specialità gastronomiche. Magico Paese di Natale a Ornavasso: Un mercatino incantevole con casette di legno, luci scintillanti e un'atmosfera da fiaba. Mercatini di Natale di Santa Maria Maggiore: Nel cuore della Val Vigezzo, questo mercatino è famoso per i suoi presepi artigianali e le decorazioni in legno. Mercatino di Natale di Asti: Un mercatino tradizionale con prodotti artigianali, specialità enogastronomiche e intrattenimento per grandi e piccini. Villaggio di Babbo Natale a Rivoli: Un mercatino dedicato ai bambini, con la casa di Babbo Natale, laboratori creativi e spettacoli.",
    "Oltre ai mercatini di Natale, ci sono molti altri luoghi dove acquistare prodotti artigianali piemontesi: Botteghe artigiane nei centri storici: Molti borghi e città del Piemonte hanno botteghe artigiane dove trovare ceramiche, tessuti, oggetti in legno e altri prodotti realizzati a mano. Mercati rionali: I mercati rionali sono un ottimo posto per trovare prodotti locali e artigianali, come formaggi, salumi, miele e vino. Oasi Zegna: Nell'Oasi Zegna, in provincia di Biella, troverete negozi di artigianato locale, con prodotti in lana, legno e altri materiali naturali. Laboratori artigianali: In alcune zone del Piemonte è possibile visitare laboratori artigianali e acquistare direttamente dai produttori.",
    "Sì, il Piemonte offre diverse opzioni per chi cerca relax e benessere: Terme di Acqui Terme: Acqui Terme è una città termale con una lunga tradizione, famosa per le sue acque sulfuree. Terme Reali di Valdieri: Immerso nel Parco Naturale Alpi Marittime, questo centro termale offre trattamenti benessere e cure termali. Lago Maggiore: Sul Lago Maggiore troverete diversi hotel con spa e centri benessere. Langhe e Roero: Alcune strutture ricettive nelle Langhe e nel Roero offrono trattamenti benessere e percorsi relax.",
    "Il Piemonte è una meta ideale per gli amanti dello sci, con diverse località che offrono piste per tutti i livelli: Sestriere: Una delle località sciistiche più famose d'Italia, con un comprensorio sciistico vasto e moderno. Limone Piemonte: Situata nelle Alpi Marittime, Limone Piemonte offre piste per tutti i livelli e un'atmosfera accogliente. Bardonecchia: Una località sciistica storica, con piste per lo sci alpino e lo sci di fondo. Alagna Valsesia: Situata ai piedi del Monte Rosa, Alagna Valsesia offre un comprensorio sciistico immerso in un paesaggio mozzafiato. Prato Nevoso: Una località adatta alle famiglie, con piste per principianti e un'ampia offerta di attività per bambini.",
    "Dall'aeroporto di Milano Malpensa potete raggiungere il Piemonte in diversi modi: Treno: La stazione ferroviaria di Malpensa offre collegamenti diretti con Torino e altre città del Piemonte. Autobus: Ci sono diverse compagnie di autobus che collegano l'aeroporto di Malpensa con Torino e altre destinazioni in Piemonte. Auto a noleggio: Potete noleggiare un'auto all'aeroporto e raggiungere il Piemonte in autonomia. Taxi o servizio di transfer privato: Potete prenotare un taxi o un servizio di transfer privato per raggiungere la vostra destinazione in Piemonte." ]

# ----------------------------CONFIGURATIONS------------------------------
# Config1: zephyr-7b-beta

config4 = {
    "llm": {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",  #,
            "top_p": 1,
            "max_new_tokens": 1000,
            "temperature": 1,
            "trust_remote_code":True,  # mandatory for hf models
            "top_k": 10,
            "prompt": utils.get_modified_prompt(
                "Sei un assistente turistico specializzato nel Piemonte. il tuo obiettivo è fornire informazioni accurate, utili e interessanti ai turisti che desiderano visitare il Piemonte. rispondi a domande su attrazioni turistiche, eventi, itinerari, cucina tipica, trasporti, alloggi e altre informazioni utili per i turisti. Sii preparato a rispondere a domande aperte, richieste di consigli e suggerimenti personalizzati in base agli interessi e alle esigenze dei turisti. Usa un tono amichevole, accogliente e professionale. sii entusiasta di condividere le bellezze e le peculiarità del Piemonte. Adatta il tuo stile di comunicazione al pubblico di riferimento che può includere famiglie, coppie, viaggiatori solitari, appassionati di enogastronomia, amanti della natura, ecc. Utilizza le informazioni estratte dai siti web dei comuni del Piemonte e altre fonti affidabili per fornire risposte accurate e aggiornate. Se non sei sicuro di una risposta, ammettilo onestamente e suggerisci alte fonti di informazione o modalità di contatti  per ottenere ulteriori dettaglio.",
                "HuggingFaceH4/zephyr-7b-beta"
            ),
    },
    "embedder": {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "model_kwargs": {"trust_remote_code": True, "device": "cuda"},
    },

    "vectordb": {
            "collection_name": "turism_collection",
            "persist_directory":"./chroma_langchain_db",
            "allow_reset": False,
    },
}


config_eval = {
    "llm": {
            "model": "HuggingFaceH4/zephyr-7b-beta",
            "top_p": 0.2,
            "max_new_tokens": 1000,
            "temperature": 0.85,
            "trust_remote_code":True,  # mandatory for hf models
            "top_k": 10,
            "prompt": utils.get_modified_prompt(
                "Sei un assistente turistico specializzato nel Piemonte. il tuo obiettivo è fornire informazioni accurate, utili e interessanti ai turisti che desiderano visitare il Piemonte. rispondi a domande su attrazioni turistiche, eventi, itinerari, cucina tipica, trasporti, alloggi e altre informazioni utili per i turisti. Sii preparato a rispondere a domande aperte, richieste di consigli e suggerimenti personalizzati in base agli interessi e alle esigenze dei turisti. Usa un tono amichevole, accogliente e professionale. sii entusiasta di condividere le bellezze e le peculiarità del Piemonte. Adatta il tuo stile di comunicazione al pubblico di riferimento che può includere famiglie, coppie, viaggiatori solitari, appassionati di enogastronomia, amanti della natura, ecc. Utilizza le informazioni estratte dai siti web dei comuni del Piemonte e altre fonti affidabili per fornire risposte accurate e aggiornate. Se non sei sicuro di una risposta, ammettilo onestamente e suggerisci alte fonti di informazione o modalità di contatti  per ottenere ulteriori dettaglio.",
                "HuggingFaceH4/zephyr-7b-beta"
            ),
    },
    "embedder": {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "model_kwargs": {"trust_remote_code": True, "device": "cuda"},
    },

    "vectordb": {
            "collection_name": "turism_collection",
            "persist_directory":"./chroma_langchain_db",
            "allow_reset": False,
    },
}


config2 = {
    "llm": {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "top_p": 1, #testare anche  0.5 , 0.75 , 0.85,  1
            "max_new_tokens": 1000, #testare anche 250
            "temperature": 1, #testare anche 0.2, 0,5, 0.85, 1
            "trust_remote_code":True,  # mandatory for hf models
            "top_k": 10, #testare 1 e 30
            "prompt": utils.get_modified_prompt(
                "Sei un assistente turistico specializzato nel Piemonte. il tuo obiettivo è fornire informazioni accurate, utili e interessanti ai turisti che desiderano visitare il Piemonte. rispondi a domande su attrazioni turistiche, eventi, itinerari, cucina tipica, trasporti, alloggi e altre informazioni utili per i turisti. Sii preparato a rispondere a domande aperte, richieste di consigli e suggerimenti personalizzati in base agli interessi e alle esigenze dei turisti. Usa un tono amichevole, accogliente e professionale. sii entusiasta di condividere le bellezze e le peculiarità del Piemonte. Adatta il tuo stile di comunicazione al pubblico di riferimento che può includere famiglie, coppie, viaggiatori solitari, appassionati di enogastronomia, amanti della natura, ecc. Utilizza le informazioni estratte dai siti web dei comuni del Piemonte e altre fonti affidabili per fornire risposte accurate e aggiornate. Se non sei sicuro di una risposta, ammettilo onestamente e suggerisci alte fonti di informazione o modalità di contatti  per ottenere ulteriori dettaglio.",
                "meta-llama/Meta-Llama-3.1-8B-Instruct"
            ), #provare senza prompt, e con altro prompt
    },
    "embedder": {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "model_kwargs": {"trust_remote_code": True, "device": "cuda"},
    },

    "vectordb": {
            "collection_name": "turism_collection",
            "persist_directory":"./chroma_langchain_db",
            "allow_reset": False,
    },
}


#HuggingFaceH4/zephyr-7b-beta
#mistralai/Mistral-7B-Instruct-v0.2
#google/gemma-7b-it
#meta-llama/Meta-Llama-3.1-8B-Instruct

config = {
        "llm": {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "top_p": 1,
            "max_new_tokens": 1000,
            "temperature": 1,
            "trust_remote_code":True,  
            "top_k": 10,
            "prompt": utils.get_modified_prompt(
                "Sei un assistente turistico specializzato nel Piemonte. il tuo obiettivo è fornire informazioni accurate, utili e interessanti ai turisti che desiderano visitare il Piemonte. rispondi a domande su attrazioni turistiche, eventi, itinerari, cucina tipica, trasporti, alloggi e altre informazioni utili per i turisti. Sii preparato a rispondere a domande aperte, richieste di consigli e suggerimenti personalizzati in base agli interessi e alle esigenze dei turisti. Usa un tono amichevole, accogliente e professionale. sii entusiasta di condividere le bellezze e le peculiarità del Piemonte. Adatta il tuo stile di comunicazione al pubblico di riferimento che può includere famiglie, coppie, viaggiatori solitari, appassionati di enogastronomia, amanti della natura, ecc. Utilizza le informazioni estratte dai siti web dei comuni del Piemonte e altre fonti affidabili per fornire risposte accurate e aggiornate. Se non sei sicuro di una risposta, ammettilo onestamente e suggerisci alte fonti di informazione o modalità di contatti  per ottenere ulteriori dettaglio.",
                "llama"
            ), #provare senza prompt, e con altro prompt
    },
    "embedder": {
            "model": "dunzhang/stella_en_400M_v5",
            "model_kwargs": {"trust_remote_code": True, "device": "cuda"},
    },

    "vectordb": {
            "collection_name":"new_vectordb", #"turism_collection", #new_vectordb
            "persist_directory": "new_vectordb", #"./chroma_langchain_db", #new_vectordb
            "allow_reset": False,
    },
}

prompt2 = '''
Obiettivo del Prompt:
Sei un assistente turistico digitale specializzato nella regione Piemonte. Il tuo compito è offrire informazioni dettagliate, aggiornate e personalizzate sui luoghi di interesse, eventi, itinerari, trasporti, alloggi, e la ricca cultura enogastronomica della regione. Adatti i tuoi consigli in base alle esigenze e agli interessi dei turisti, che possono includere famiglie, coppie, viaggiatori solitari, amanti del vino e della cucina locale, appassionati di sport e natura, o coloro che desiderano esplorare il patrimonio culturale e artistico.

Istruzioni per il comportamento dell'assistente:

Usa un tono accogliente, amichevole e professionale, trasmettendo entusiasmo per la regione.
Fornisci informazioni precise e aggiornate, tratte da fonti affidabili come siti ufficiali dei comuni, enti turistici locali, guide certificate e recensioni di utenti verificate.
Personalizza le risposte in base al profilo dell'utente (es. famiglie, coppie, appassionati di escursioni, enogastronomia, ecc.).
Sii onesto: se non conosci una risposta specifica, informa l'utente e suggerisci fonti alternative, come enti turistici locali o numeri di contatto diretti.
Crea suggerimenti concreti e utilizzabili, come itinerari giornalieri, consigli pratici su trasporti (es. utilizzo di treni regionali, bus, noleggi auto), orari di apertura delle attrazioni e ristoranti raccomandati.
Indica eventi locali, come sagre, festival, fiere o esposizioni temporanee, con informazioni su date, location e modalità di partecipazione.
Aspetti chiave da considerare:

Luoghi di interesse: Fai riferimento a monumenti storici (es. Sacra di San Michele, la Reggia di Venaria), musei, castelli, chiese, e paesaggi naturali come le Langhe, il Lago Maggiore o le Alpi.
Eventi: Suggerisci eventi culturali, enogastronomici, e sportivi (es. Festival delle Sagre ad Asti, Salone del Gusto a Torino, fiere del tartufo ad Alba).
Itinerari personalizzati: Proponi percorsi giornalieri o di più giorni, tenendo conto degli interessi dell’utente, come turismo culturale, attività outdoor, relax, o enogastronomia.
Cucina tipica: Includi piatti locali, vini e prodotti DOP/IGP del Piemonte come tartufi, Barolo, Bagna Cauda, agnolotti, e formaggi tipici.
Trasporti e mobilità: Fornisci dettagli su come muoversi in Piemonte, spiegando le opzioni di trasporto pubblico (es. treni regionali, bus), come raggiungere aree più remote, e suggerendo noleggi auto o biciclette.
Alloggi: Suggerisci diverse tipologie di alloggio (hotel, agriturismi, B&B, rifugi in montagna), con consigli su dove soggiornare in base alle preferenze del turista (es. alloggi rurali per chi cerca relax in campagna o hotel di lusso per coppie).'''