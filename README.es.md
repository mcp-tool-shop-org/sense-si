<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Versión:** 1.2.0

Servidor MCP para la evaluación visual basada en datos. Proporciona a Claude un juicio honesto sobre las imágenes mediante SigLIP2; *mide*, no narra, por lo que no alucina.

## El problema

Cuando Claude necesita verificar qué hay en una imagen ("¿este sprite tiene una espada?", "¿hay un botón de inicio de sesión?"), los VLMs generativos (LLaVA, GPT-4V) alucinan respuestas convincentes. Completan narrativas, no observaciones. LLaVA 13B informó que "el personaje sostiene una gran espada" en imágenes donde no había ningún arma, con una confianza del 90%, en cada recorte.

## La solución

SigLIP2 es un modelo de visión discriminativo. No genera texto; mide la similitud entre una imagen y una descripción textual, devolviendo una puntuación sigmoide calibrada. Cuando el arma está presente, la puntuación es 10 a 100 veces mayor que cuando no lo está. Cuando no puede determinarlo, la puntuación es baja. No alucina.

Este servidor MCP encapsula SigLIP2 como herramientas que cualquier flujo de trabajo de Claude puede llamar.

## Herramientas

| Herramienta | Qué hace |
|------|-------------|
| `image_contains` | "¿Esta imagen contiene X?" → puntuación sigmoide |
| `image_classify` | Puntuar la imagen en función de N etiquetas candidatas |
| `image_compare` | Similitud coseno entre dos imágenes, en comparación con un "umbral diferente" proporcionado por el llamante |
| `image_score_batch` | Puntuar N imágenes en relación con una consulta |
| `image_verify` | Juicio RELATIVO honesto: objetivo frente a alternativas → decisión + margen + confianza |
| `image_rank` | Clasificar N candidatos en relación con una referencia → los k mejores con márgenes |
| `eyes_selftest` | Autoprueba en imágenes de referencia incluidas (demuestra la instalación y la calibración) |
| `eyes_status` | Comprobación de estado: modelo, dispositivo, estado cargado |

### Cuándo recurrir a otra cosa

ai-eyes responde a la pregunta **"¿es verdadera esta afirmación sobre los píxeles?"**. Pondera una hipótesis que usted proporciona; no puede decirle qué hay en una imagen que aún no ha descrito.

Para un **título, una descripción o texto extraído de la imagen**, se trata de un trabajo generativo y requiere una herramienta diferente: **[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)** (Florence-2). Su resultado puede alucinar detalles por construcción; vuelva a traer cualquier elemento importante aquí y mídalo con `image_verify`.

Un par deliberado, y la guía de cada uno apunta al otro: **plain-sight describe, ai-eyes mide.**

## Primeros pasos

```bash
pip install -e .
ai-eyes-mcp  # starts STDIO server
```

O ejecute como un módulo: `python -m ai_eyes_mcp`

### Configuración de Claude Code

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

## Configuración

| Variable de entorno | Valor predeterminado | Propósito |
|---------|---------|---------|
| `AI_EYES_MODEL_ID` | `google/siglip2-so400m-patch14-384` | Modelo HuggingFace |
| `AI_EYES_MODEL_REVISION` | SHA del commit fijado | Revisión del modelo. **Debe ser un SHA de commit hexadecimal de 40 caracteres.** `main`, una etiqueta o un valor vacío provocan un fallo grave al cargar, no una alternativa; consulte *Reproducibilidad* a continuación. |
| `AI_EYES_MODEL_DIR` | Caché predeterminada de HF | Directorio de caché del modelo |
| `AI_EYES_DEVICE` | `cuda` si está disponible, de lo contrario `cpu` | Dispositivo torch. Establezca un dispositivo literal (`cuda`, `cpu`, `cuda:1`); no hay ningún valor `auto`; `AI_EYES_DEVICE=auto` genera un error. |
| `AI_EYES_DEFAULT_THRESHOLD` | `0.02` | Umbral predeterminado para `image_contains` |
| `AI_EYES_LOG_LEVEL` | `WARNING` | Verbosidad del registro: `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `AI_EYES_EAGER_LOAD` | no establecido | Si es verdadero, cargue el modelo al inicio para que un modelo/caché defectuoso falle rápidamente (no en la primera llamada a la herramienta) |
| `AI_EYES_DTYPE` | precisión total | `float16` / `bfloat16` para reducir a la mitad la VRAM |
| `AI_EYES_EMBED_CACHE` | `64` | Tamaño de la memoria de incrustación de imágenes en memoria. Se basa en la ruta + mtime + tamaño, por lo que un archivo reescrito se vuelve a medir y nunca se sirve una versión obsoleta. No hay disco ni archivo secundario. |

### Reproducibilidad: los pesos están fijados

Una puntuación solo tiene sentido si sabe qué pesos la produjeron. La revisión del modelo está **fijada a un SHA de commit específico**, que se pasa a cada carga y se **informa en cada carga útil que contiene un número**. Dos instalaciones realizadas con meses de diferencia devuelven la misma puntuación para la misma entrada.

Dejar una revisión sin establecer se resuelve al valor fijado, nunca a una rama flotante. Un reemplazo solo se respeta como un SHA diferente de 40 caracteres (intención del operador); `main`, una etiqueta o una cadena vacía hacen que la carga falle con un mensaje procesable en lugar de desviarse silenciosamente.

**Registro:** el servidor registra los datos en el registrador `ai_eyes_mcp` a **stderr** (stdout es el canal del protocolo MCP). Establezca el nivel con `AI_EYES_LOG_LEVEL` (arriba) o adjunte sus propios controladores a `logging.getLogger("ai_eyes_mcp")`.

**Primera llamada:** el modelo se carga de forma diferida; la **primera** llamada a la herramienta de imagen descarga/carga SigLIP2 (~10–20 segundos en GPU; más tiempo en la primera descarga) y las llamadas posteriores son de ~100 ms. Establezca `AI_EYES_EAGER_LOAD=1` para cargar al inicio del servidor o llame a `eyes_status`, que informa sobre `loaded` sin activar una carga. **No es gratuito en un servidor inactivo**: la primera llamada implica una importación de biblioteca única (se mide en ~10 segundos; ~2 ms una vez que está listo).

## Cómo funcionan las puntuaciones

SigLIP2 utiliza la puntuación **sigmoide**, no softmax. Cada par imagen-texto obtiene una probabilidad independiente (0-1):

Solo ilustración aproximada; estas bandas no se transfieren entre consultas o estilos de imagen, y la siguiente sección explica por qué no debe basarse en ellas:

- **Alto** (>0.1): fuerte coincidencia visual
- **Bajo** (<0.01): coincidencia débil o nula
- **Medio** (0.01-0.1): ambiguo

Las puntuaciones NO son relativas. Múltiples consultas pueden obtener una puntuación alta en la misma imagen (por ejemplo, una imagen con una espada y un escudo).

### ⚠ La formulación de la consulta es importante; prefiera `image_classify` para tomar decisiones sólidas

Las puntuaciones sigmoides de SigLIP2 son **sensibles a la formulación de la consulta**: la puntuación absoluta para la *misma* imagen varía mucho según la redacción (una frase que coincida con el estilo puede obtener una puntuación 10 a 100 veces mayor que una genérica). Por lo tanto, un valor fijo de `threshold` requiere una ingeniería de consultas por caso de uso y los umbrales no se transfieren entre estilos de imagen.

Para tomar decisiones robustas de sí/no con diversas entradas, prefiera **`image_classify`**: *clasifica* las etiquetas candidatas entre sí y no es sensible a la magnitud absoluta de la puntuación. Utilice `image_contains` con un umbral ajustado solo cuando controle tanto la redacción de la consulta como el estilo de la imagen. `eyes_status` refleja esto en su campo `scoring_guidance`.

El umbral predeterminado (`0.02`) es un límite permisivo, no un corte universal; ajústelo para sus consultas y el estilo de la imagen o utilice `image_classify`.

## Arquitectura

```
engine.py          Standalone SigLIP2 wrapper — no MCP dependency.
                   Lazy-loads model on first inference call.
                   Importable directly for non-MCP use cases.

server.py          FastMCP wrapper that exposes engine methods as MCP tools.
                   Thin layer: input validation, error shaping, tool metadata.

__main__.py        Entry point for `python -m ai_eyes_mcp`.
```

`engine.py` es el núcleo: gestiona la carga del modelo, la selección del dispositivo y toda la lógica de inferencia. `server.py` nunca interactúa directamente con torch; delega todo al motor. Esto significa que puede usar `from ai_eyes_mcp.engine import SigLIPEngine` y utilizarlo en cualquier script de Python sin incluir FastMCP.

ai-eyes-mcp evalúa imágenes y nada más. No tiene ninguna opinión sobre lo que haga con el número: catalogación, flujos de trabajo de sprites, puertas de enlace de CI en los recursos generados; eso corresponde al consumidor.

## Referencia de la herramienta

### `image_contains`

```
image_contains(image_path, query, threshold=0.02)
```

Comprueba si una imagen contiene algo descrito por la consulta. Devuelve una puntuación sigmoide independiente (de 0 a 1).

| Parámetro | Tipo | Obligatorio | Descripción |
|-----------|------|----------|-------------|
| `image_path` | string | sí | Ruta absoluta al archivo de imagen |
| `query` | string | sí | Qué buscar (por ejemplo, "una persona sosteniendo una espada") |
| `threshold` | float | no | Umbral de puntuación para la decisión positiva (predeterminado: 0.02) |

Devuelve: `{present, score, threshold, query, truncated, revision, elapsed_ms}`

`truncated: true` significa que su consulta superó la capacidad de 64 tokens del codificador de texto y **la puntuación refleja solo los primeros 64 tokens**: trátelo como incompleto, no como un número.

### `image_classify`

```
image_classify(image_path, labels)
```

Asigna una puntuación a una imagen en comparación con varias etiquetas candidatas. Devuelve puntuaciones sigmoides independientes; NO softmax.

| Parámetro | Tipo | Obligatorio | Descripción |
|-----------|------|----------|-------------|
| `image_path` | string | sí | Ruta absoluta al archivo de imagen |
| `labels` | string[] | sí | Etiquetas candidatas para asignarles una puntuación (máximo 20) |

Devuelve: `{scores, best, best_score, truncated, revision, elapsed_ms}`

`best` se selecciona a partir de las puntuaciones de precisión completa; los valores mostrados solo se redondean hasta el punto en que sean coherentes con esa elección.

### `image_compare`

```
image_compare(image_a, image_b, baselines=None)
```

Calcula la similitud visual entre dos imágenes utilizando la similitud del coseno de las incrustaciones SigLIP2.

| Parámetro | Tipo | Obligatorio | Descripción |
|-----------|------|----------|-------------|
| `image_a` | string | sí | Ruta absoluta a la primera imagen |
| `image_b` | string | sí | Ruta absoluta a la segunda imagen |
| `baselines` | string[][] | no | Pares de imágenes que **no** coinciden en su estilo. A-B solo se considera separado si supera este umbral. |

**Sin `baselines`, el número es una medida, no una decisión**: la carga útil contiene `incomplete: true`. Los umbrales de similitud no se transfieren entre los estilos de imagen, por lo que la herramienta no inventará uno. Se mide en seis pares de *diferentes* personajes en un estilo de sprite: **0.698–0.836**, frente a 1.0 para una imagen comparada consigo misma. Un corte fijo elegido a partir de eso sería incorrecto para fotos, capturas de pantalla o renders; por lo tanto, usted proporciona el contraste, exactamente como `image_verify` acepta alternativas.

Devuelve: `{similarity, separated, incomplete, margin, baseline_max, confidence, image_a, image_b, revision, elapsed_ms}`

### `image_score_batch`

```
image_score_batch(image_paths, query, threshold=0.02)
```

Asigna una puntuación a varias imágenes en comparación con una sola consulta. Máximo 100 imágenes por llamada.

| Parámetro | Tipo | Obligatorio | Descripción |
|-----------|------|----------|-------------|
| `image_paths` | string[] | sí | Lista de rutas absolutas de las imágenes |
| `query` | string | sí | Qué buscar |
| `threshold` | float | no | Umbral de puntuación (predeterminado: 0.02) |

Devuelve: `{query, threshold, total, scored, present, absent, errors, error_details?, results: [{path, score, present}], truncated, revision, elapsed_ms}`

### `image_verify`

```
image_verify(image_path, target, alternatives)
```

Decisión **relativa** honesta: clasifica `target` en comparación con `alternatives` proporcionada por el llamador (obligatorio, ≥1) y devuelve una decisión + margen + confianza. Es robusto a la sensibilidad de SigLIP a la redacción de las consultas porque es relativo, no un umbral absoluto. Para obtener una puntuación sin procesar, utilice `image_contains`; para obtener una clasificación completa, utilice `image_classify`.

| Parámetro | Tipo | Obligatorio | Descripción |
|-----------|------|----------|-------------|
| `image_path` | string | sí | Ruta absoluta a la imagen |
| `target` | string | sí | La hipótesis que se va a verificar |
| `alternatives` | string[] | sí | Alternativas de contraste con las que clasificar (≥1) |

Devuelve: `{present, target, target_score, best_alternative, best_alternative_score, margin, confidence, truncated, revision, elapsed_ms}`; `confidence` es `high` / `moderate` / `low — inconclusive`, lo que describe la diferencia medida.

### `image_rank`

```
image_rank(reference, candidates, k=5, baselines=None)
```

Clasifica los candidatos por similitud con una referencia. Codifica la referencia **una sola vez** en lugar de por cada candidato.

| Parámetro | Tipo | Obligatorio | Descripción |
|-----------|------|----------|-------------|
| `reference` | string | sí | Ruta absoluta a la imagen de referencia |
| `candidates` | string[] | sí | Rutas de las imágenes candidatas para clasificarlas |
| `k` | int | no | Número máximo de coincidencias que se devolverán (predeterminado: 5) |
| `baselines` | string[][] | no | Pares que **no** coinciden en su estilo; el umbral por debajo del cual nada es "cercano". |

Devuelve: `{matches, nothing_close, incomplete, baseline_max, k, reference, revision, elapsed_ms}`

**Sin `baselines`, la clasificación es una medida, no "estas coinciden"** (`incomplete: true`). Con valores de referencia, solo los candidatos que superen el umbral proporcionado por el llamador son coincidencias; y si ninguno lo supera, `matches` estará **vacío**. Una función de clasificación que siempre devuelve k resultados es una respuesta segura en un entorno sin señal; esta se niega.

### `eyes_status`

```
eyes_status()
```

Comprueba el estado del servidor. No activa la carga del modelo.

Devuelve: `{model_id, revision, device, dtype, loaded, cache_dir, parameters?, vram_mb?, scoring_guidance, note?}`

Cuando `loaded` es verdadero, también devuelve `parameters` (`1136M`) y `vram_mb` (solo CUDA).

### `eyes_selftest`

```
eyes_selftest()
```

Ejecuta el modelo en algunas imágenes de referencia incluidas y confirma que se mantienen los ordenamientos esperados; demuestra que la instalación se cargó correctamente y que SigLIP2 está calibrado. Carga el modelo si aún no se ha cargado.

Devuelve: `{passed, checks: [{name, expected, measured_a, measured_b, ok}], model_id, device, torch_version, transformers_version}`

`measured_a` / `measured_b` se informan con toda la resolución: una puntuación pequeña distinta de cero se muestra en notación científica (por ejemplo, `4.3813e-08`) en lugar de reducirse a `0`.

## Seguridad y confianza

Esta herramienta funciona **solo localmente**.

- **Datos accedidos:** archivos de imagen locales (solo lectura), caché del modelo HuggingFace (descargada una vez)
- **Red:** toda la *inferencia* es local; no hay salida, nunca. La única excepción es la **primera ejecución**, que descarga aproximadamente 1.6 GB de pesos del modelo desde HuggingFace. Después de eso, la herramienta ya no se conecta a la red.
- **Para un despliegue aislado o con auditoría de salida:** prepopule la caché y apunte `AI_EYES_MODEL_DIR` hacia ella. La herramienta entonces no realizará ninguna llamada de red en ningún momento.
- **No gestiona secretos**: no lee, almacena ni transmite credenciales ni claves API
- **No hay telemetría**: no se recopila ni envía nada
- **No modifica archivos**: los archivos de imagen se abren en modo de solo lectura, nunca se modifican
- **No realiza acciones peligrosas**: no hay operaciones de eliminación, finalización o reinicio
- **Solo errores estructurados**: las trazas de pila nunca se exponen a los clientes

## Requisitos

- Python >= 3.10
- Se recomienda una GPU CUDA. **VRAM medida: ~4,3 GB con la configuración predeterminada `float32`**, ~2,2 GB con `AI_EYES_DTYPE=float16`. El modelo tiene 1136 millones de parámetros; "SO400M" es el nombre de la torre de visión, no del modelo completo.
- Disponible una opción alternativa para CPU (más lenta, aproximadamente 10 veces más).
- La descarga del modelo ocupa ~1,6 GB en el primer uso.

## Desarrollo

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

Utilice el **script de consola** `pytest`, no `python -m pytest`. El segundo coloca el directorio de trabajo en `sys.path` y puede ocultar un error de importación que la integración continua detectará.

## Licencia

MIT

---

Creado por [MCP Tool Shop](https://mcp-tool-shop.github.io/)
