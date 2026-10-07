<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Version :** 1.2.0

Serveur MCP pour l’évaluation visuelle basée sur des données réelles. Permet à Claude de formuler un jugement honnête sur les images grâce à SigLIP2 — il *mesure*, il ne décrit pas, donc il n’invente rien.

## Le problème

Lorsque Claude doit vérifier le contenu d’une image (« cet élément contient-il une épée ? », « y a-t-il un bouton de connexion ?»), les modèles VLM génératifs (LLaVA, GPT-4V) produisent des réponses affirmatives mais erronées. Ils complètent des récits plutôt que de faire des observations. LLaVA 13B a indiqué « le personnage tient une grande épée » dans des images où il n’y avait aucune arme, avec un niveau de confiance de 0,90, pour chaque recadrage.

## La solution

SigLIP2 est un modèle de vision discriminatif. Il ne génère pas de texte ; il mesure la similarité entre une image et une description textuelle, en renvoyant un score sigmoïde calibré. Lorsque l’arme est présente, le score est 10 à 100 fois plus élevé que lorsqu’elle est absente. Lorsqu’il ne peut pas déterminer, le score est faible. Il n’invente rien.

Ce serveur MCP encapsule SigLIP2 en tant qu’outils auxquels tout flux de travail Claude peut faire appel.

## Outils

| Outil | Fonctionnement |
|------|-------------|
| `image_contains` | « Cette image contient-elle X ? » → score sigmoïde |
| `image_classify` | Attribuer un score à l’image par rapport à N étiquettes candidates |
| `image_compare` | Similarité cosinus entre deux images, par rapport à une valeur minimale « différente » fournie par l’appelant |
| `image_score_batch` | Attribuer un score à N images par rapport à une requête |
| `image_verify` | Verdict HONNÊTE et RELATIF : cible par rapport aux alternatives → décision + marge + niveau de confiance |
| `image_rank` | Classer N candidats par rapport à une référence → top-k avec marges |
| `eyes_selftest` | Autotest sur les images de référence incluses (prouve l’installation et la calibration) |
| `eyes_status` | Vérification de l’état : modèle, appareil, état chargé |

### Quand utiliser autre chose

ai-eyes répond à la question « Cette affirmation concernant les pixels est-elle vraie ? ». Il attribue un score à une hypothèse que vous fournissez — il ne peut pas vous dire ce qui se trouve dans une image que vous n’avez pas déjà décrite.

Pour une légende, une description ou un texte extrait de l’image, il s’agit d’un travail génératif et d’un outil différent : **[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)** (Florence-2). Sa sortie peut inventer des détails par construction — renvoyez tout élément essentiel ici et mesurez-le avec `image_verify`.

Un ensemble délibéré, et les conseils de chacun pointent vers l’autre : **plain-sight décrit, ai-eyes mesure.**

## Démarrage rapide

```bash
pip install -e .
ai-eyes-mcp  # starts STDIO server
```

Ou exécuter en tant que module : `python -m ai_eyes_mcp`

### Configuration de Claude Code

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

## Configuration

| Variable d’environnement | Valeur par défaut | Objectif |
|---------|---------|---------|
| `AI_EYES_MODEL_ID` | `google/siglip2-so400m-patch14-384` | Modèle HuggingFace |
| `AI_EYES_MODEL_REVISION` | SHA de validation du commit | Révision du modèle. **Doit être un SHA de commit hexadécimal à 40 caractères.** `main`, une balise ou une valeur vide entraînent un échec de chargement irréversible, et non une solution de repli — voir *Reproductibilité* ci-dessous. |
| `AI_EYES_MODEL_DIR` | Cache par défaut HF | Répertoire du cache du modèle |
| `AI_EYES_DEVICE` | `cuda` si disponible, sinon `cpu` | Appareil torch. Définir un appareil littéral (`cuda`, `cpu`, `cuda:1`) — il n’y a pas de valeur `auto` ; `AI_EYES_DEVICE=auto` provoque une erreur. |
| `AI_EYES_DEFAULT_THRESHOLD` | `0.02` | Seuil par défaut pour `image_contains` |
| `AI_EYES_LOG_LEVEL` | `WARNING` | Verbosité des journaux : `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `AI_EYES_EAGER_LOAD` | non défini | Si cette valeur est vraie, charger le modèle au démarrage afin qu’un modèle/cache défectueux échoue rapidement (et non lors du premier appel de l’outil). |
| `AI_EYES_DTYPE` | Précision complète | `float16` / `bfloat16` pour réduire de moitié la VRAM |
| `AI_EYES_EMBED_CACHE` | `64` | Taille de la mémoire en RAM pour les embeddings d’images. Clé basée sur le chemin + mtime + taille, de sorte qu’un fichier réécrit est remesuré et n’est jamais servi dans une version obsolète. Pas de disque, pas de fichier annexe. |

### Reproductibilité — les poids sont fixés

Un score n’a de sens que si vous savez quels poids l’ont produit. La révision du modèle est **fixée à un SHA de commit spécifique**, transmise à chaque chargement, et **indiquée dans chaque charge utile contenant un nombre**. Deux installations effectuées à plusieurs mois d’intervalle renvoient le même score pour la même entrée.

Laisser une révision non définie revient à utiliser la valeur fixée — jamais une branche flottante. Un remplacement n’est pris en compte que comme un SHA différent de 40 caractères (intention de l’opérateur) ; `main`, une balise ou une chaîne vide entraînent un échec du chargement avec un message clair plutôt qu’une dérive silencieuse.

**Journalisation :** le serveur enregistre les informations dans le journal `ai_eyes_mcp` vers **stderr** (stdout est le canal de protocole MCP). Définir le niveau avec `AI_EYES_LOG_LEVEL` (ci-dessus) ou attacher vos propres gestionnaires à `logging.getLogger("ai_eyes_mcp")`.

**Premier appel :** le modèle se charge en mode paresseux — le **premier** appel d’outil image télécharge/charge SigLIP2 (~10 à 20 secondes sur GPU ; plus long lors du premier téléchargement), et les appels suivants sont d’environ 100 ms. Définir `AI_EYES_EAGER_LOAD=1` pour charger au démarrage du serveur, ou appeler `eyes_status`, qui signale `loaded` sans déclencher de chargement. **Ce n’est pas gratuit sur un serveur froid** — le premier appel implique un coût unique d’importation de la bibliothèque (mesuré à environ 10 secondes ; ~2 ms une fois que le système est opérationnel).

## Fonctionnement des scores

SigLIP2 utilise un score **sigmoïde**, et non softmax. Chaque paire image-texte reçoit une probabilité indépendante (de 0 à 1) :

Illustration approximative uniquement — **ces plages ne sont pas transférables entre les requêtes ou les styles d’image**, et la section suivante explique pourquoi vous ne devez pas vous y fier :

- **Élevé** (>0,1) : forte correspondance visuelle
- **Faible** (<0,01) : faible ou aucune correspondance
- **Moyen** (0,01 à 0,1) : ambigu

Les scores ne sont PAS relatifs. Plusieurs requêtes peuvent obtenir un score élevé pour la même image (par exemple, une image contenant à la fois une épée et un bouclier).

### ⚠ La formulation de la requête est importante — privilégiez `image_classify` pour des décisions plus robustes

Les scores sigmoïdes SigLIP2 sont **sensibles à la formulation de la requête** : le score absolu pour la *même* image varie considérablement en fonction du libellé (une phrase adaptée au style peut obtenir un score 10 à 100 fois plus élevé qu’une phrase générique). Par conséquent, une valeur fixe `threshold` nécessite une conception de requête spécifique à chaque cas d’utilisation, et les seuils ne sont pas transférables entre les styles d’image.

Pour prendre des décisions robustes du type oui/non en fonction de différents types de données, privilégiez **`image_classify`** ; il *classe* les étiquettes candidates les unes par rapport aux autres et n’est pas sensible à l’amplitude absolue des scores. Utilisez `image_contains` avec un seuil ajusté uniquement lorsque vous contrôlez à la fois la formulation de la requête et le style de l’image. `eyes_status` reprend ce principe dans son champ `scoring_guidance`.

La valeur par défaut du seuil (`0.02`) représente une limite inférieure tolérante et non une valeur absolue universelle ; ajustez-la en fonction de vos requêtes et du style de l’image, ou utilisez `image_classify`.

## Architecture

```
engine.py          Standalone SigLIP2 wrapper — no MCP dependency.
                   Lazy-loads model on first inference call.
                   Importable directly for non-MCP use cases.

server.py          FastMCP wrapper that exposes engine methods as MCP tools.
                   Thin layer: input validation, error shaping, tool metadata.

__main__.py        Entry point for `python -m ai_eyes_mcp`.
```

`engine.py` constitue le cœur du système : il gère le chargement des modèles, la sélection des appareils et toute la logique d’inférence. `server.py` n’interagit jamais directement avec Torch ; il délègue toutes les tâches au moteur. Cela signifie que vous pouvez utiliser `from ai_eyes_mcp.engine import SigLIPEngine` et l’intégrer dans n’importe quel script Python sans avoir à importer FastMCP.

«ai-eyes-mcp » évalue uniquement les images et rien d’autre. Il n’a aucune opinion sur l’utilisation que vous faites des données obtenues (catalogage, création de séquences d’images animées, intégration dans des processus d’intégration continue pour les éléments générés), car cela relève de la responsabilité de l’utilisateur final.

## Référence de l’outil

### `image_contains`

```
image_contains(image_path, query, threshold=0.02)
```

Vérifiez si une image contient un élément correspondant à la description fournie dans la requête. Renvoie un score sigmoïde indépendant (de 0 à 1).

| Paramètre | Type | Requis | Description |
|-----------|------|----------|-------------|
| `image_path` | chaîne de caractères | oui | Chemin d’accès absolu au fichier image. |
| `query` | chaîne de caractères | oui | Quels éléments rechercher (par exemple, « une personne tenant une épée »). |
| `threshold` | flotter ; nombre à virgule flottante | no | Seuil de score pour le résultat actuel (valeur par défaut : 0,02). |

Retourne : `{present, score, threshold, query, truncated, revision, elapsed_ms}`

`truncated: true` signifie que votre requête a dépassé la capacité de 64 jetons du codeur de texte et que **le score ne reflète que les 64 premiers jetons**. Considérez-le comme un résultat incomplet, et non comme une valeur numérique.

### `image_classify`

```
image_classify(image_path, labels)
```

Attribuez un score à une image en fonction de plusieurs étiquettes candidates. Renvoie des scores sigmoïdes indépendants, et non des scores obtenus par la fonction softmax.

| Paramètre | Type | Requis | Description |
|-----------|------|----------|-------------|
| `image_path` | chaîne de caractères | oui | Chemin d’accès absolu au fichier image. |
| `labels` | tableau de chaînes de caractères | oui | Libellés des candidats à évaluer (maximum 20) |

Retourne : `{scores, best, best_score, truncated, revision, elapsed_ms}`

Les valeurs de `best` sont sélectionnées parmi les résultats obtenus avec une précision maximale ; les valeurs affichées sont arrondies uniquement dans la mesure nécessaire pour qu’elles soient cohérentes avec ce choix.

### `image_compare`

```
image_compare(image_a, image_b, baselines=None)
```

Calculez la similarité visuelle entre deux images en utilisant la similarité cosinus des plongements SigLIP2.

| Paramètre | Type | Requis | Description |
|-----------|------|----------|-------------|
| `image_a` | chaîne de caractères | oui | Chemin d’accès absolu à la première image. |
| `image_b` | chaîne de caractères | oui | Chemin d’accès absolu à la deuxième image. |
| `baselines` | tableau de tableaux de chaînes de caractères | no | Les paires d’images qui ne correspondent pas à votre style sont considérées comme distinctes uniquement si leur nombre dépasse ce seuil minimal. |

**Sans `baselines`, le nombre représente une mesure et non un jugement** – la charge utile contient `incomplete: true`. Les seuils de similarité ne sont pas transposables d’un style d’image à l’autre, donc l’outil n’en inventera pas un. La mesure est effectuée sur six paires de caractères *différents* dans un même style de sprite : **0,698–0,836**, contre 1,0 pour une image comparée à elle-même. Un seuil fixe choisi parmi ces valeurs serait inapproprié pour les photos, les captures d’écran ou les rendus – vous devez donc fournir le contraste, exactement comme `image_verify` gère les alternatives.

Retourne : `{similarity, separated, incomplete, margin, baseline_max, confidence, image_a, image_b, revision, elapsed_ms}`

### `image_score_batch`

```
image_score_batch(image_paths, query, threshold=0.02)
```

Évaluez plusieurs images en fonction d’une seule requête. Nombre maximal de 100 images par appel.

| Paramètre | Type | Requis | Description |
|-----------|------|----------|-------------|
| `image_paths` | tableau de chaînes de caractères | oui | Liste des chemins d’accès absolus aux images. |
| `query` | chaîne de caractères | oui | Quels sont les éléments à prendre en compte ? |
| `threshold` | flotter ; nombre à virgule flottante | no | Seuil de score (valeur par défaut : 0,02) |

Retourne : `{query, threshold, total, scored, present, absent, errors, error_details?, results: [{path, score, present}], truncated, revision, elapsed_ms}`

### `image_verify`

```
image_verify(image_path, target, alternatives)
```

Verdict **relatif** et objectif : compare la valeur `target` à la valeur `alternatives` fournie par l’utilisateur (obligatoire, ≥1) et renvoie une décision, une marge et un niveau de confiance. Résistant à la sensibilité de SigLIP en matière de formulation des requêtes, car il s’agit d’une comparaison relative et non d’un seuil absolu. Pour obtenir un score brut, utilisez `image_contains` ; pour obtenir un classement complet, utilisez `image_classify`.

| Paramètre | Type | Requis | Description |
|-----------|------|----------|-------------|
| `image_path` | chaîne de caractères | oui | Chemin d’accès absolu à l’image. |
| `target` | chaîne de caractères | oui | L’hypothèse à vérifier. |
| `alternatives` | tableau de chaînes de caractères | oui | Indiquez les options à comparer pour établir un classement (au moins une). |

Retourne : `{present, target, target_score, best_alternative, best_alternative_score, margin, confidence, truncated, revision, elapsed_ms}` – `confidence` est égal à `high` / `moderate` / `low — inconclusive`, ce qui décrit l’écart mesuré.

### `image_rank`

```
image_rank(reference, candidates, k=5, baselines=None)
```

Classez les candidats en fonction de leur similarité avec un élément de référence. L’élément de référence est encodé une seule fois, et non pour chaque candidat.

| Paramètre | Type | Requis | Description |
|-----------|------|----------|-------------|
| `reference` | chaîne de caractères | oui | Chemin d’accès absolu à l’image de référence. |
| `candidates` | tableau de chaînes de caractères | oui | Chemins d’accès aux images des candidats à classer. |
| `k` | entier | no | Nombre maximal de correspondances à afficher (par défaut : 5). |
| `baselines` | tableau de tableaux de chaînes de caractères | no | Les associations qui ne correspondent pas à votre style : en dessous d’un certain seuil, aucune combinaison n’est acceptable. |

Retourne : `{matches, nothing_close, incomplete, baseline_max, k, reference, revision, elapsed_ms}`

**Sans `baselines`, le classement est une simple mesure, et non une indication de « correspondance »** (`incomplete: true`). Avec des valeurs de référence, seules les correspondances qui dépassent le seuil défini par l’utilisateur sont prises en compte, et si aucune ne dépasse ce seuil, `matches` est **vide**. Un verbe de classement qui renvoie toujours k résultats constitue une réponse fiable dans un contexte où il n’y a pas de signal ; celui-ci refuse.

### `eyes_status`

```
eyes_status()
```

Vérifiez l’état du serveur. Cette action ne déclenche pas le chargement du modèle.

Retourne : `{model_id, revision, device, dtype, loaded, cache_dir, parameters?, vram_mb?, scoring_guidance, note?}`

Lorsque `loaded` est vrai, il renvoie également `parameters` (`1136M`) et `vram_mb` (uniquement pour CUDA).

### `eyes_selftest`

```
eyes_selftest()
```

Exécute le modèle sur un ensemble d’images de référence et vérifie que l’ordre attendu est respecté, ce qui prouve que l’installation s’est déroulée correctement et que SigLIP2 a été calibré. Charge le modèle s’il n’a pas déjà été chargé.

Retourne : `{passed, checks: [{name, expected, measured_a, measured_b, ok}], model_id, device, torch_version, transformers_version}`

Les valeurs `measured_a` et `measured_b` sont affichées avec la résolution maximale ; une valeur minuscule, mais non nulle, est présentée sous forme de notation scientifique (par exemple, `4.3813e-08`) plutôt que d’être arrondie à `0`.

## Sécurité et confiance

Cet outil fonctionne **uniquement en mode local**.

- **Données concernées :** fichiers d’images locaux (en lecture seule), cache du modèle HuggingFace (téléchargé une seule fois)
- **Réseau :** toutes les *inférences* sont locales, il n’y a jamais de communication sortante. La seule exception est la **première exécution**, qui télécharge environ 1,6 Go de poids du modèle depuis HuggingFace. Après cela, l’outil n’accède plus au réseau.
- **Pour un déploiement dans un environnement isolé ou avec audit des communications sortantes :** préchargez le cache et indiquez à `AI_EYES_MODEL_DIR` où il se trouve. L’outil n’effectuera alors aucune communication réseau à aucun moment.
- **Aucune gestion de données sensibles** : ne lit, ne stocke ni ne transmet pas d’identifiants ou de clés API.
- **Aucune télémétrie** : rien n’est collecté ni envoyé.
- **Aucune modification de fichier** : les fichiers d’images sont ouverts en lecture seule et ne sont jamais modifiés.
- **Aucune action dangereuse** : aucune opération de suppression, d’arrêt ou de redémarrage.
- **Seulement des erreurs structurées** : les traces de pile ne sont jamais exposées aux clients.

## Exigences

- Python >= 3.10
- Une carte graphique CUDA est recommandée. **Mémoire VRAM mesurée : environ 4,3 Go avec la configuration par défaut `float32`**, environ 2,2 Go avec `AI_EYES_DTYPE=float16`. Le modèle comporte 1 136 millions de paramètres — « SO400M » désigne le module de vision, et non le modèle assemblé.
- Une solution de repli sur CPU est disponible (plus lente, environ 10 fois plus lente).
- Le téléchargement du modèle représente environ 1,6 Go lors de la première utilisation.

## Développement

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

Utilisez le **script de console** `pytest`, et non `python -m pytest`. Ce dernier place le répertoire de travail sur `sys.path` et peut masquer un échec d’importation que l’intégration continue (CI) détectera.

## Licence

MIT

---

Créé par [MCP Tool Shop](https://mcp-tool-shop.github.io/)
