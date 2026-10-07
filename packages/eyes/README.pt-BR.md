<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

**Versão:** 1.2.0

Servidor MCP para avaliação visual baseada em dados. Fornece a Claude uma análise honesta de imagens por meio do SigLIP2 — ele *mede*, não narra, portanto, não alucina.

## O Problema

Quando Claude precisa verificar o que está em uma imagem — “este sprite tem uma espada?”, “existe um botão de login?” — os VLMs generativos (LLaVA, GPT-4V) alucinam respostas confiantes. Eles completam narrativas, não fazem observações. O LLaVA 13B relatou “o personagem está segurando uma espada grande” em imagens onde não havia nenhuma arma, com uma confiança de 0,90, em cada recorte.

## A Solução

SigLIP2 é um modelo de visão discriminativo. Ele não gera texto — mede a similaridade entre uma imagem e uma descrição textual, retornando uma pontuação sigmoide calibrada. Quando a arma está presente, a pontuação é 10 a 100 vezes maior do que quando está ausente. Quando ele não consegue determinar, a pontuação é baixa. Ele não alucina.

Este servidor MCP envolve o SigLIP2 como ferramentas que qualquer fluxo de trabalho do Claude pode chamar.

## Ferramentas

| Ferramenta | O que ela faz |
|------|-------------|
| `image_contains` | “Esta imagem contém X?” → pontuação sigmoide |
| `image_classify` | Pontuar a imagem em relação a N rótulos candidatos |
| `image_compare` | Similaridade de cosseno entre duas imagens, em relação a um limite “diferente” fornecido pelo chamador |
| `image_score_batch` | Pontuar N imagens em relação a uma consulta |
| `image_verify` | Avaliação RELATIVA honesta: alvo versus alternativas → decisão + margem + confiança |
| `image_rank` | Classificar N candidatos em relação a uma referência → top-k com margens |
| `eyes_selftest` | Autoteste em imagens de referência incluídas (comprova a instalação + calibração) |
| `eyes_status` | Verificação de integridade: modelo, dispositivo, estado carregado |

### Quando usar outra coisa

ai-eyes responde **“esta afirmação sobre os pixels é verdadeira?”** Ele pontua uma hipótese que você fornece — ele não pode dizer o que há em uma imagem que você ainda não descreveu.

Para uma **legenda, descrição ou texto extraído da imagem**, esse é um trabalho generativo e um instrumento diferente: **[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)** (Florence-2). Sua saída pode alucinar detalhes por construção — traga qualquer coisa que suporte a carga de volta aqui e meça com `image_verify`.

Um par deliberado, e a orientação de cada um aponta para o outro: **plain-sight descreve, ai-eyes mede.**

## Primeiros Passos

```bash
pip install -e .
ai-eyes-mcp  # starts STDIO server
```

Ou execute como um módulo: `python -m ai_eyes_mcp`

### Configuração do Claude Code

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

## Configuração

| Variável de ambiente | Padrão | Finalidade |
|---------|---------|---------|
| `AI_EYES_MODEL_ID` | `google/siglip2-so400m-patch14-384` | Modelo HuggingFace |
| `AI_EYES_MODEL_REVISION` | SHA do commit fixado | Revisão do modelo. **Deve ser um SHA de commit hexadecimal de 40 caracteres.** `main`, uma tag ou um valor vazio são falhas graves no carregamento, não uma alternativa — veja *Reprodutibilidade* abaixo. |
| `AI_EYES_MODEL_DIR` | Cache padrão do HF | Diretório de cache do modelo |
| `AI_EYES_DEVICE` | `cuda` se disponível, caso contrário `cpu` | Dispositivo torch. Defina um dispositivo literal (`cuda`, `cpu`, `cuda:1`) — não há valor `auto`; `AI_EYES_DEVICE=auto` gera um erro. |
| `AI_EYES_DEFAULT_THRESHOLD` | `0.02` | Limite padrão para `image_contains` |
| `AI_EYES_LOG_LEVEL` | `WARNING` | Verbose do log: `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `AI_EYES_EAGER_LOAD` | não definido | Se verdadeiro, carregue o modelo na inicialização para que um modelo/cache com problemas falhe rapidamente (e não na primeira chamada da ferramenta) |
| `AI_EYES_DTYPE` | Precisão total | `float16` / `bfloat16` para reduzir pela metade a VRAM |
| `AI_EYES_EMBED_CACHE` | `64` | Tamanho do memo de incorporação de imagem em memória. Chaveado por caminho + mtime + tamanho, portanto, um arquivo reescrito é medido novamente, nunca servido com dados desatualizados. Sem disco, sem arquivo auxiliar. |

### Reprodutibilidade — os pesos são fixados

Uma pontuação só tem significado se você souber quais pesos a produziram. A revisão do modelo é **fixada em um SHA de commit específico**, passada para cada carregamento e **relatada em cada carga útil que contém um número**. Duas instalações feitas com meses de diferença retornam a mesma pontuação para a mesma entrada.

Deixar uma revisão não definida resolve para o fixo — nunca para um branch flutuante. Uma substituição é honrada apenas como um SHA diferente de 40 caracteres (intenção do operador); `main`, uma tag ou uma string vazia fazem com que o carregamento falhe com uma mensagem acionável em vez de desviar silenciosamente.

**Registro:** O servidor registra sob o logger `ai_eyes_mcp` para **stderr** (stdout é o canal de protocolo MCP). Defina o nível com `AI_EYES_LOG_LEVEL` (acima) ou anexe seus próprios manipuladores a `logging.getLogger("ai_eyes_mcp")`.

**Primeira chamada:** o modelo carrega de forma preguiçosa — a **primeira** chamada da ferramenta de imagem baixa/carrega o SigLIP2 (~10–20 segundos na GPU; mais longo no primeiro download) e as chamadas subsequentes são ~100 ms. Defina `AI_EYES_EAGER_LOAD=1` para carregar na inicialização do servidor ou chame `eyes_status`, que relata `loaded` sem acionar um carregamento. **Não é gratuito em um servidor inativo** — a primeira chamada tem um custo único de importação da biblioteca (medido ~10 segundos; ~2 ms quando está pronto).

## Como as Pontuações Funcionam

O SigLIP2 usa pontuação **sigmoide**, não softmax. Cada par imagem-texto recebe uma probabilidade independente (0-1):

Apenas ilustração aproximada — **essas faixas não são transferidas entre consultas ou estilos de imagem** e a próxima seção explica por que você não deve construí-las:

- **Alto** (>0,1): forte correspondência visual
- **Baixo** (<0,01): correspondência fraca ou nenhuma
- **Médio** (0,01–0,1): ambíguo

As pontuações NÃO são relativas. Várias consultas podem obter pontuações altas na mesma imagem (por exemplo, uma imagem com uma espada e um escudo).

### ⚠ A formulação da consulta é importante — prefira `image_classify` para decisões robustas

As pontuações sigmoid do SigLIP2 são **sensíveis à formulação da consulta**: a pontuação absoluta para a *mesma* imagem varia amplamente com a redação (uma frase correspondente ao estilo pode obter uma pontuação 10–100 vezes maior do que uma genérica). Uma `threshold` fixa, portanto, precisa de engenharia de consultas por caso de uso e os limites não são transferidos entre estilos de imagem.

Para decisões robustas de sim/não em diversas entradas, prefira **`image_classify`** — ele *classifica* os rótulos candidatos uns contra os outros e é insensível à magnitude absoluta da pontuação. Recorra a `image_contains` com um limite ajustado apenas quando você controlar tanto a formulação da consulta quanto o estilo da imagem. `eyes_status` ecoa isso em seu campo `scoring_guidance`.

O limite padrão (`0.02`) é um valor mínimo permissivo, não um corte universal — ajuste-o para suas consultas e estilo de imagem ou use `image_classify`.

## Arquitetura

```
engine.py          Standalone SigLIP2 wrapper — no MCP dependency.
                   Lazy-loads model on first inference call.
                   Importable directly for non-MCP use cases.

server.py          FastMCP wrapper that exposes engine methods as MCP tools.
                   Thin layer: input validation, error shaping, tool metadata.

__main__.py        Entry point for `python -m ai_eyes_mcp`.
```

`engine.py` é o núcleo — ele gerencia o carregamento do modelo, a seleção do dispositivo e toda a lógica de inferência. `server.py` nunca interage diretamente com o PyTorch; ele delega tudo para o motor. Isso significa que você pode usar `from ai_eyes_mcp.engine import SigLIPEngine` e usá-lo em qualquer script Python sem importar o FastMCP.

ai-eyes-mcp avalia imagens e nada mais. Não tem opinião sobre o que você faz com o número — catalogação, pipelines de sprites, portões de CI em ativos gerados — isso cabe ao consumidor.

## Referência da ferramenta

### `image_contains`

```
image_contains(image_path, query, threshold=0.02)
```

Verifica se uma imagem contém algo descrito pela consulta. Retorna uma pontuação sigmoide independente (0-1).

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|----------|-------------|
| `image_path` | string | sim | Caminho absoluto para o arquivo de imagem |
| `query` | string | sim | O que procurar (por exemplo, "uma pessoa segurando uma espada") |
| `threshold` | float | no | Limite de pontuação para a decisão positiva (padrão 0,02) |

Retorna: `{present, score, threshold, query, truncated, revision, elapsed_ms}`

`truncated: true` significa que sua consulta excedeu a capacidade de 64 tokens do codificador de texto e **a pontuação reflete apenas os primeiros 64 tokens** — trate-o como incompleto, não como um número.

### `image_classify`

```
image_classify(image_path, labels)
```

Atribui uma pontuação a uma imagem em relação a vários rótulos candidatos. Retorna pontuações sigmoidais independentes — NÃO softmax.

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|----------|-------------|
| `image_path` | string | sim | Caminho absoluto para o arquivo de imagem |
| `labels` | string[] | sim | Rótulos candidatos para os quais atribuir uma pontuação (máximo de 20) |

Retorna: `{scores, best, best_score, truncated, revision, elapsed_ms}`

`best` é selecionado a partir de pontuações de precisão total; os valores exibidos são arredondados apenas até o ponto em que os mantêm consistentes com essa escolha.

### `image_compare`

```
image_compare(image_a, image_b, baselines=None)
```

Calcula a similaridade visual entre duas imagens usando a similaridade do cosseno dos embeddings SigLIP2.

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|----------|-------------|
| `image_a` | string | sim | Caminho absoluto para a primeira imagem |
| `image_b` | string | sim | Caminho absoluto para a segunda imagem |
| `baselines` | string[][] | no | Pares de imagens que **não** correspondem ao seu estilo. A-B é contado como separado apenas se exceder esse limite mínimo. |

**Sem `baselines`, o número é uma medida, não uma decisão** — a carga útil carrega `incomplete: true`. Os limites de similaridade não são transferidos entre os estilos de imagem, portanto, a ferramenta não inventará um. Medido em seis pares de *diferentes* personagens em um estilo de sprite: **0,698–0,836**, em comparação com 1,0 para uma imagem contra si mesma. Um limite fixo escolhido a partir disso seria incorreto para fotos, capturas de tela ou renderizações — portanto, você fornece o contraste, exatamente como `image_verify` aceita alternativas.

Retorna: `{similarity, separated, incomplete, margin, baseline_max, confidence, image_a, image_b, revision, elapsed_ms}`

### `image_score_batch`

```
image_score_batch(image_paths, query, threshold=0.02)
```

Atribui pontuações a várias imagens em relação a uma única consulta. Máximo de 100 imagens por chamada.

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|----------|-------------|
| `image_paths` | string[] | sim | Lista de caminhos absolutos das imagens |
| `query` | string | sim | O que procurar |
| `threshold` | float | no | Limite de pontuação (padrão 0,02) |

Retorna: `{query, threshold, total, scored, present, absent, errors, error_details?, results: [{path, score, present}], truncated, revision, elapsed_ms}`

### `image_verify`

```
image_verify(image_path, target, alternatives)
```

Decisão **relativa** honesta — classifica `target` em relação ao `alternatives` fornecido pelo chamador (obrigatório, ≥1) e retorna uma decisão + margem + confiança. Robusta à sensibilidade da formulação da consulta do SigLIP porque é relativa, não um limite absoluto. Para uma pontuação bruta, use `image_contains`; para classificação completa, use `image_classify`.

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|----------|-------------|
| `image_path` | string | sim | Caminho absoluto para a imagem |
| `target` | string | sim | A hipótese a ser verificada |
| `alternatives` | string[] | sim | Alternativas de contraste para classificar (≥1) |

Retorna: `{present, target, target_score, best_alternative, best_alternative_score, margin, confidence, truncated, revision, elapsed_ms}` — `confidence` é `high` / `moderate` / `low — inconclusive`, descrevendo a diferença medida.

### `image_rank`

```
image_rank(reference, candidates, k=5, baselines=None)
```

Classifica os candidatos por similaridade com uma referência. Codifica a referência **uma vez**, em vez de por candidato.

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|----------|-------------|
| `reference` | string | sim | Caminho absoluto para a imagem de referência |
| `candidates` | string[] | sim | Caminhos das imagens candidatas para classificar |
| `k` | int | no | Número máximo de correspondências a serem retornadas (padrão 5) |
| `baselines` | string[][] | no | Pares que **não** correspondem ao seu estilo — o limite abaixo do qual nada é "próximo" |

Retorna: `{matches, nothing_close, incomplete, baseline_max, k, reference, revision, elapsed_ms}`

**Sem `baselines`, a classificação é uma medida, não "estas correspondem"** (`incomplete: true`). Com linhas de base, apenas os candidatos acima do limite fornecido pelo chamador são correspondências — e se nenhum deles atingir esse limite, `matches` estará **vazio**. Uma função de classificação que sempre retorna k resultados é uma resposta confiante em um regime sem sinal; esta recusa.

### `eyes_status`

```
eyes_status()
```

Verifica o status do servidor. Não aciona o carregamento do modelo.

Retorna: `{model_id, revision, device, dtype, loaded, cache_dir, parameters?, vram_mb?, scoring_guidance, note?}`

Quando `loaded` é verdadeiro, também retorna `parameters` (`1136M`) e `vram_mb` (somente CUDA).

### `eyes_selftest`

```
eyes_selftest()
```

Executa o modelo em algumas imagens de referência agrupadas e confirma que as ordenações esperadas são mantidas — prova que a instalação foi carregada corretamente e que o SigLIP2 está calibrado. Carrega o modelo se ele já não estiver carregado.

Retorna: `{passed, checks: [{name, expected, measured_a, measured_b, ok}], model_id, device, torch_version, transformers_version}`

`measured_a` / `measured_b` são relatados em resolução total — uma pontuação pequena e diferente de zero é renderizada em notação científica (por exemplo, `4.3813e-08`) em vez de ser reduzida a `0`.

## Segurança e Confiança

Esta ferramenta opera **apenas localmente**.

- **Dados acessados:** arquivos de imagem locais (somente leitura), cache do modelo HuggingFace (baixado uma vez)
- **Rede:** toda a *inferência* é local — sem saída, nunca. A única exceção é a **primeira execução**, que baixa cerca de 1,6 GB de pesos do modelo do HuggingFace. Depois disso, a ferramenta nunca acessa a rede novamente.
- **Para uma implantação com isolamento de rede ou auditoria de saída:** pré-popule o cache e aponte `AI_EYES_MODEL_DIR` para ele. A ferramenta então não faz nenhuma chamada de rede em nenhum momento.
- **Nenhum tratamento de segredos** — não lê, armazena ou transmite credenciais ou chaves de API
- **Nenhuma telemetria** — nada é coletado ou enviado
- **Nenhuma mutação de arquivo** — os arquivos de imagem são abertos em modo somente leitura, nunca modificados
- **Nenhuma ação perigosa** — nenhuma operação de exclusão, interrupção ou reinicialização
- **Apenas erros estruturados** — rastreamentos de pilha nunca expostos aos clientes

## Requisitos

- Python >= 3.10
- GPU CUDA recomendada. **VRAM medida: ~4,3 GB com as configurações padrão `float32`**, ~2,2 GB com `AI_EYES_DTYPE=float16`. O modelo tem 1136 milhões de parâmetros — "SO400M" é o nome da torre de visão, não do modelo completo.
- Existe uma opção de recurso para CPU (mais lenta, cerca de 10 vezes).
- O download do modelo ocupa aproximadamente 1,6 GB na primeira utilização.

## Desenvolvimento

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

Utilize o **script da linha de comandos** `pytest`, e não `python -m pytest`. Este último define o diretório de trabalho em `sys.path` e pode ocultar uma falha na importação que será detetada pelo CI.

## Licença

MIT

---

Criado por [MCP Tool Shop](https://mcp-tool-shop.github.io/)
