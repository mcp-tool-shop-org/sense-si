<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="docs/logo.png" alt="ai-eyes" width="360">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/ai-eyes-mcp/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/ai-eyes-mcp/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
</p>

# ai-eyes-mcp

**Versione:** 1.2.0

Server MCP per la valutazione visiva basata su dati reali. Fornisce a Claude un giudizio onesto sulle immagini tramite SigLIP2: *misura*, non narra, quindi non "allucina".

## Il problema

Quando Claude deve verificare cosa è presente in un'immagine ("questo sprite ha una spada?", "c'è un pulsante di accesso?"), i modelli VLM generativi (LLaVA, GPT-4V) forniscono risposte sicure ma errate. Completano narrazioni, non effettuano osservazioni. LLaVA 13B ha segnalato che "il personaggio tiene una spada a due mani" in immagini in cui non era presente alcuna arma, con un livello di confidenza del 90%, per ogni singolo ritaglio.

## La soluzione

SigLIP2 è un modello di visione discriminativo. Non genera testo: misura la somiglianza tra un'immagine e una descrizione testuale, restituendo un punteggio sigmoide calibrato. Quando l'arma è presente, il punteggio è 10-100 volte superiore rispetto a quando non lo è. Quando non riesce a determinare, il punteggio è basso. Non "allucina".

Questo server MCP integra SigLIP2 come strumento che qualsiasi flusso di lavoro Claude può richiamare.

## Strumenti

| Strumento | Cosa fa |
|------|-------------|
| `image_contains` | "Questa immagine contiene X?" → punteggio sigmoide |
| `image_classify` | Valuta l'immagine rispetto a N etichette candidate |
| `image_compare` | Somiglianza del coseno tra due immagini, rispetto a una soglia "diversa" fornita dal chiamante |
| `image_score_batch` | Valuta N immagini rispetto a una singola query |
| `image_verify` | Giudizio RELATIVO onesto: target vs. alternative → decisione + margine + confidenza |
| `image_rank` | Classifica N candidati rispetto a un riferimento → top-k con margini |
| `eyes_selftest` | Autotest sulle immagini di riferimento incluse (verifica l'installazione e la calibrazione) |
| `eyes_status` | Controllo dello stato: modello, dispositivo, stato caricato |

### Quando utilizzare qualcos'altro

ai-eyes risponde alla domanda **"questa affermazione sui pixel è vera?"**. Valuta un'ipotesi che fornisci tu: non può dirti cosa c'è in un'immagine che non hai già descritto.

Per una **didascalia, una descrizione o un testo estratto dall'immagine**, si tratta di un lavoro generativo e richiede uno strumento diverso: **[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)** (Florence-2). La sua output può "allucinare" dettagli per costruzione: riporta qui qualsiasi elemento importante e misuralo con `image_verify`.

Una coppia deliberata, e le indicazioni di ciascuna puntano all'altra: **plain-sight descrive, ai-eyes misura.**

## Avvio rapido

```bash
pip install -e .
ai-eyes-mcp  # starts STDIO server
```

Oppure esegui come modulo: `python -m ai_eyes_mcp`

### Configurazione di Claude Code

```json
{
  "mcpServers": {
    "ai-eyes": {
      "command": "ai-eyes-mcp",
      "env": {
        "AI_EYES_MODEL_DIR": "/path/to/model/cache"
      }
    }
  }
}
```

## Configurazione

| Variabile d'ambiente | Predefinito | Scopo |
|---------|---------|---------|
| `AI_EYES_MODEL_ID` | `google/siglip2-so400m-patch14-384` | Modello HuggingFace |
| `AI_EYES_MODEL_REVISION` | Commit SHA fissato | Revisione del modello. **Deve essere un commit SHA esadecimale di 40 caratteri.** `main`, un tag o un valore vuoto causano un errore di caricamento, non un fallback: vedi *Riproducibilità* qui sotto. |
| `AI_EYES_MODEL_DIR` | Cache predefinita HF | Directory della cache del modello |
| `AI_EYES_DEVICE` | `cuda` se disponibile, altrimenti `cpu` | Dispositivo torch. Imposta un dispositivo letterale (`cuda`, `cpu`, `cuda:1`): non esiste un valore `auto`; `AI_EYES_DEVICE=auto` genera un errore. |
| `AI_EYES_DEFAULT_THRESHOLD` | `0.02` | Soglia predefinita per `image_contains` |
| `AI_EYES_LOG_LEVEL` | `WARNING` | Livello di dettaglio del log: `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `AI_EYES_EAGER_LOAD` | non impostato | Se è vero, carica il modello all'avvio in modo che un modello/cache difettoso fallisca rapidamente (e non alla prima chiamata dello strumento) |
| `AI_EYES_DTYPE` | precisione completa | `float16` / `bfloat16` per dimezzare la VRAM |
| `AI_EYES_EMBED_CACHE` | `64` | Dimensione della memoria in-memory per l'embedding delle immagini. Indicizzata su percorso + mtime + dimensione, quindi un file riscritto viene rimisurato e non viene mai servita una versione obsoleta. Nessun disco, nessun file secondario. |

### Riproducibilità: i pesi sono fissati

Un punteggio è significativo solo se si conoscono i pesi che lo hanno prodotto. La revisione del modello è **fissata a un commit SHA specifico**, passata a ogni caricamento e **riportata in ogni payload contenente un numero**. Due installazioni eseguite a distanza di mesi restituiscono lo stesso punteggio per lo stesso input.

Lasciare la revisione non impostata si risolve nel valore fissato, mai a una branch fluttuante. Un override viene onorato solo come SHA diverso di 40 caratteri (intento dell'operatore); `main`, un tag o una stringa vuota causano l'errore di caricamento con un messaggio utile invece che fallire silenziosamente.

**Logging:** il server registra i dati nel logger `ai_eyes_mcp` su **stderr** (stdout è il canale del protocollo MCP). Imposta il livello con `AI_EYES_LOG_LEVEL` (sopra) o aggiungi i tuoi gestori a `logging.getLogger("ai_eyes_mcp")`.

**Prima chiamata:** il modello viene caricato in modo lazy: la **prima** chiamata allo strumento immagine scarica/carica SigLIP2 (~10-20 secondi su GPU; più lungo alla prima operazione di download) e le chiamate successive sono ~100 ms. Imposta `AI_EYES_EAGER_LOAD=1` per caricare all'avvio del server o chiama `eyes_status`, che segnala `loaded` senza attivare il caricamento. **Non è gratuito su un server "freddo":** la prima chiamata comporta un costo una tantum per l'importazione della libreria (misurato ~10 secondi; ~2 ms quando è attivo).

## Come funzionano i punteggi

SigLIP2 utilizza il punteggio **sigmoide**, non softmax. Ogni coppia immagine-testo riceve una probabilità indipendente (da 0 a 1):

Solo un'illustrazione approssimativa: **queste bande non si trasferiscono tra le query o gli stili delle immagini** e la sezione successiva spiega perché non dovresti basarti su di esse:

- **Alto** (>0,1): forte corrispondenza visiva
- **Basso** (<0,01): debole o nessuna corrispondenza
- **Medio** (0,01-0,1): ambiguo

I punteggi NON sono relativi. Più query possono ottenere un punteggio alto sulla stessa immagine (ad esempio, un'immagine con una spada e uno scudo).

### ⚠ La formulazione della query è importante: preferisci `image_classify` per decisioni più affidabili

I punteggi sigmoidali di SigLIP2 sono **sensibili alla formulazione della query**: il punteggio assoluto per la *stessa* immagine varia notevolmente in base alla formulazione (una frase che corrisponde allo stile può ottenere un punteggio 10-100 volte superiore rispetto a una generica). Pertanto, un valore fisso `threshold` richiede l'ottimizzazione delle query per ogni caso d'uso e le soglie non si trasferiscono tra gli stili delle immagini.

Per prendere decisioni affidabili del tipo sì/no con input diversi, è preferibile utilizzare **`image_classify`**: esso *ordina* le etichette candidate confrontandole tra loro ed è insensibile all’entità assoluta dei punteggi. Si può ricorrere a `image_contains` solo quando si controllano sia la formulazione della query che lo stile dell’immagine, impostando una soglia appropriata. `eyes_status` riflette questo concetto nel suo campo `scoring_guidance`.

La soglia predefinita (`0.02`) rappresenta un valore minimo consentito, non una soglia universale: è possibile regolarla in base alle proprie esigenze e allo stile delle immagini oppure utilizzare `image_classify`.

## Architettura

```
engine.py          Standalone SigLIP2 wrapper — no MCP dependency.
                   Lazy-loads model on first inference call.
                   Importable directly for non-MCP use cases.

server.py          FastMCP wrapper that exposes engine methods as MCP tools.
                   Thin layer: input validation, error shaping, tool metadata.

__main__.py        Entry point for `python -m ai_eyes_mcp`.
```

`engine.py` rappresenta il nucleo centrale: gestisce il caricamento del modello, la selezione del dispositivo e tutta la logica di inferenza. `server.py` non interagisce mai direttamente con Torch; delega tutte le operazioni al motore. Ciò significa che è possibile utilizzare `from ai_eyes_mcp.engine import SigLIPEngine` in qualsiasi script Python senza dover includere FastMCP.

AI-Eyes-MCP analizza le immagini e nient’altro. Non esprime alcun parere su come utilizzi i dati numerici ottenuti – ad esempio, per la catalogazione, nelle pipeline di animazione o nei sistemi di controllo qualità degli asset generati –, poiché tale aspetto è di competenza dell’utente finale.

## Riferimenti sugli strumenti

### `image_contains`

```
image_contains(image_path, query, threshold=0.02)
```

Verifica se un’immagine contiene elementi descritti nella richiesta. Restituisce un punteggio indipendente, espresso tramite una funzione sigmoide (da 0 a 1).

| Parametro | Tipo | Obbligatorio/a | Descrizione |
|-----------|------|----------|-------------|
| `image_path` | stringa | sì | Percorso assoluto del file immagine. |
| `query` | stringa | sì | Cosa cercare (ad esempio, «una persona che tiene una spada»). |
| `threshold` | galleggiare; numero in virgola mobile | no | Soglia di punteggio per la convalida del risultato attuale (valore predefinito: 0,02). |

Restituisce: `{present, score, threshold, query, truncated, revision, elapsed_ms}`

`truncated: true` indica che la tua richiesta ha superato il limite di 64 token del codificatore di testo e **il punteggio riflette solo i primi 64 token**: considera quindi il risultato come incompleto, non come un valore numerico definitivo.

### `image_classify`

```
image_classify(image_path, labels)
```

Confronta un’immagine con diverse etichette possibili e restituisci i relativi punteggi indipendenti ottenuti tramite la funzione sigmoide; non vengono utilizzati punteggi normalizzati tramite la funzione softmax.

| Parametro | Tipo | Obbligatorio/a | Descrizione |
|-----------|------|----------|-------------|
| `image_path` | stringa | sì | Percorso assoluto del file immagine. |
| `labels` | array di stringhe | sì | Etichette dei candidati da valutare (massimo 20) |

Restituisce: `{scores, best, best_score, truncated, revision, elapsed_ms}`

I valori di `best` vengono selezionati tra i punteggi calcolati con la massima precisione; i valori visualizzati sono arrotondati solo fino al punto necessario per mantenerli coerenti con tale scelta.

### `image_compare`

```
image_compare(image_a, image_b, baselines=None)
```

Calcola la somiglianza visiva tra due immagini utilizzando la similarità coseno degli embedding di SigLIP2.

| Parametro | Tipo | Obbligatorio/a | Descrizione |
|-----------|------|----------|-------------|
| `image_a` | stringa | sì | Percorso assoluto della prima immagine. |
| `image_b` | stringa | sì | Percorso assoluto della seconda immagine. |
| `baselines` | array di array di stringhe | no | Coppie di immagini che non corrispondono allo stile desiderato. La coppia A-B viene considerata separata solo se la differenza supera una certa soglia minima. |

**Senza `baselines`, il numero rappresenta una misurazione, non un giudizio**, e i dati contengono `incomplete: true`. I valori di somiglianza minima non vengono trasferiti tra diversi stili di immagine, quindi lo strumento non ne creerà uno automaticamente. La misurazione è stata effettuata su sei coppie di *caratteri diversi* all’interno dello stesso stile grafico: **0,698–0,836**, rispetto a 1,0 quando si confronta un’immagine con se stessa. Un valore limite fisso scelto in base a questi dati sarebbe errato per foto, screenshot o rendering; pertanto, è necessario fornire il contrasto, esattamente come fa `image_verify` con le alternative.

Restituisce: `{similarity, separated, incomplete, margin, baseline_max, confidence, image_a, image_b, revision, elapsed_ms}`

### `image_score_batch`

```
image_score_batch(image_paths, query, threshold=0.02)
```

È possibile confrontare più immagini con una singola richiesta. Il numero massimo di immagini per ogni operazione è 100.

| Parametro | Tipo | Obbligatorio/a | Descrizione |
|-----------|------|----------|-------------|
| `image_paths` | array di stringhe | sì | Elenco dei percorsi assoluti delle immagini. |
| `query` | stringa | sì | A cosa prestare attenzione. |
| `threshold` | galleggiare; numero in virgola mobile | no | Soglia del punteggio (valore predefinito: 0,02) |

Restituisce: `{query, threshold, total, scored, present, absent, errors, error_details?, results: [{path, score, present}], truncated, revision, elapsed_ms}`

### `image_verify`

```
image_verify(image_path, target, alternatives)
```

Valutazione **relativa** e imparziale: confronta il valore `target` con il valore `alternatives` fornito dall’utente (obbligatorio, ≥1) e restituisce una decisione, un margine di errore e un livello di confidenza. È resistente alla sensibilità alle formulazioni delle query di SigLIP perché si basa su una valutazione relativa e non su una soglia assoluta. Per ottenere un punteggio grezzo, utilizzare `image_contains`; per una classifica completa, utilizzare `image_classify`.

| Parametro | Tipo | Obbligatorio/a | Descrizione |
|-----------|------|----------|-------------|
| `image_path` | stringa | sì | Percorso assoluto dell’immagine. |
| `target` | stringa | sì | L’ipotesi da verificare. |
| `alternatives` | array di stringhe | sì | Indica le alternative da confrontare per stabilire una graduatoria (almeno 1). |

Restituisce: `{present, target, target_score, best_alternative, best_alternative_score, margin, confidence, truncated, revision, elapsed_ms}` – `confidence` è `high` / `moderate` / `low — inconclusive`, indicando la dimensione dello spazio misurato.

### `image_rank`

```
image_rank(reference, candidates, k=5, baselines=None)
```

Classifica i candidati in base alla loro somiglianza con un elemento di riferimento. L’elemento di riferimento viene codificato una sola volta, anziché per ogni candidato.

| Parametro | Tipo | Obbligatorio/a | Descrizione |
|-----------|------|----------|-------------|
| `reference` | stringa | sì | Percorso assoluto dell’immagine di riferimento. |
| `candidates` | array di stringhe | sì | Percorsi delle immagini dei candidati da ordinare in base alla loro rilevanza. |
| `k` | int | no | Numero massimo di corrispondenze da visualizzare (valore predefinito: 5). |
| `baselines` | array di array di stringhe | no | Le combinazioni che non si adattano al tuo stile – quelle che non offrono alcuna alternativa accettabile. |

Restituisce: `{matches, nothing_close, incomplete, baseline_max, k, reference, revision, elapsed_ms}`

**Senza `baselines`, la classifica è semplicemente una misurazione, non indica quali elementi corrispondono** (`incomplete: true`). Con i valori di riferimento, solo i candidati che superano la soglia minima fornita vengono considerati corrispondenze; se nessuno la supera, `matches` sarà **vuoto**. Un verbo di classificazione che restituisce sempre k risultati è una risposta affidabile in un contesto privo di segnali; questo invece rifiuta.

### `eyes_status`

```
eyes_status()
```

Verifica lo stato del server. Questa operazione non avvia il caricamento del modello.

Restituisce: `{model_id, revision, device, dtype, loaded, cache_dir, parameters?, vram_mb?, scoring_guidance, note?}`

Quando `loaded` è vero, restituisce anche `parameters` (`1136M`) e `vram_mb` (solo per CUDA).

### `eyes_selftest`

```
eyes_selftest()
```

Esegue il modello su un insieme di immagini di riferimento e verifica che l’ordine previsto sia corretto, dimostrando così che l’installazione è stata completata correttamente e che SigLIP2 è stato calibrato. Carica il modello se non è già stato caricato.

Restituisce: `{passed, checks: [{name, expected, measured_a, measured_b, ok}], model_id, device, torch_version, transformers_version}`

I valori `measured_a` e `measured_b` vengono visualizzati con la massima risoluzione; anche un valore minimo, diverso da zero, viene rappresentato in notazione scientifica (ad esempio, `4.3813e-08`) anziché essere arrotondato a `0`.

## Sicurezza e affidabilità

Questo strumento funziona esclusivamente in modalità locale.

- **Dati interessati:** file di immagini locali (in sola lettura), cache del modello HuggingFace (scaricata una sola volta)
- **Rete:** tutte le operazioni di *inferenza* vengono eseguite in locale, senza alcuna comunicazione verso l’esterno. L’unica eccezione è la **prima esecuzione**, durante la quale vengono scaricati circa 1,6 GB di dati del modello da HuggingFace. Successivamente, lo strumento non effettuerà più alcuna connessione alla rete.
- **Per un ambiente isolato o con controllo delle comunicazioni in uscita:** precaricare la cache e indirizzare `AI_EYES_MODEL_DIR` verso di essa. In questo modo, lo strumento non effettuerà alcuna chiamata di rete in nessun momento.
- **Nessuna gestione di dati sensibili:** non legge, memorizza o trasmette credenziali o chiavi API.
- **Nessun telemetria:** non vengono raccolti né inviati dati.
- **Nessuna modifica dei file:** i file immagine vengono aperti in sola lettura e mai modificati.
- **Nessuna operazione pericolosa:** non vengono eseguite operazioni di eliminazione, interruzione o riavvio.
- **Solo errori strutturati:** le tracce dello stack non vengono mai esposte ai client.

## Requisiti

- Python >= 3.10
- Si consiglia una GPU CUDA. **VRAM misurata: circa 4,3 GB con le impostazioni predefinite `float32`**, circa 2,2 GB con `AI_EYES_DTYPE=float16`. Il modello ha 1136 milioni di parametri; "SO400M" si riferisce alla parte dedicata alla visione, non al modello completo.
- È disponibile un fallback per la CPU (più lento, circa 10 volte).
- Al primo utilizzo, il modello richiede il download di circa 1,6 GB.

## Sviluppo

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run CI-safe tests (no model required) — this is what CI runs
pytest -m "not dogfood"

# Run everything, including GPU tests that need the model
pytest

# Full verify: imports, tool registration, cold-status gate, wheel build, CI-safe tests
bash verify.sh
```

Utilizzare lo **script della console** `pytest`, non `python -m pytest`. Quest'ultimo imposta la directory di lavoro su `sys.path` e potrebbe nascondere un errore di importazione che il sistema di integrazione continua (CI) rileverà.

## Licenza

MIT

---

Realizzato da [MCP Tool Shop](https://mcp-tool-shop.github.io/)
