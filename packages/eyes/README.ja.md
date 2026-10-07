<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**バージョン:** 1.2.0

画像認識による評価を行うMCPサーバー。SigLIP2を使用して、Claudeに対して客観的な画像判断を提供します。これは単に「測定」するだけであり、物語を語るわけではないため、誤った情報を生成することはありません。

## 問題点

Claudeが画像の要素を確認する必要がある場合（「このスプライトは剣を持っているか？」「ログインボタンはあるか？」）、生成型VLMs（LLaVA、GPT-4V）は自信のある誤った回答を生成することがあります。これらは観察ではなく、物語を完成させようとします。LLaVA 13Bは、武器が存在しない画像に対して、「キャラクターがグレートソードを持っている」と0.90の確度で報告し、すべての切り出し画像で同様の結果を示しました。

## 解決策

SigLIP2は識別型の視覚モデルです。テキストを生成するのではなく、画像とテキスト記述間の類似性を測定し、調整されたシグモイドスコアを返します。武器が存在する場合、スコアは存在しない場合よりも10〜100倍高くなります。判断できない場合は、スコアは低くなります。誤った情報を生成することはありません。

このMCPサーバーは、SigLIP2をツールとしてラップし、Claudeのワークフローから呼び出すことができます。

## ツール

| ツール | 機能 |
|------|-------------|
| `image_contains` | 「この画像にはXが含まれていますか？」→シグモイドスコア |
| `image_classify` | N個の候補ラベルに対して画像を評価 |
| `image_compare` | 2つの画像間のコサイン類似度を、呼び出し元が指定した「異なる」基準値と比較 |
| `image_score_batch` | 1つのクエリに対してN個の画像を評価 |
| `image_verify` | 客観的な相対的判断：ターゲット vs 別の選択肢 → 決定 + マージン + 確度 |
| `image_rank` | 1つの参照画像に対してN個の候補をランク付け → 上位k件とマージン |
| `eyes_selftest` | バンドルされた参照画像に対する自己テスト（インストールとキャリブレーションを確認） |
| `eyes_status` | ヘルスチェック：モデル、デバイス、ロード状態 |

### 別のものを検討すべき場合

ai-eyesは「このピクセルに関する主張は正しいですか？」に答えます。提供された仮説を評価します。まだ説明していない画像の要素を判断することはできません。

**キャプション、説明、または画像から読み取ったテキスト**の場合は、それは生成的な作業であり、別のツールです：**[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)**（Florence-2）。その出力は、構造的に詳細を誤って生成する可能性があります。重要な要素については、ここで測定し、`image_verify`を使用してください。

意図的にペアになっているツールであり、それぞれのツールの機能は互いを補完します：**plain-sightは記述し、ai-eyesは測定します。**

## クイックスタート

```bash
pip install -e .
ai-eyes-mcp  # starts STDIO server
```

または、モジュールとして実行することもできます：`python -m ai_eyes_mcp`

### Claude Code設定

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

## 構成

| 環境変数 | デフォルト値 | 目的 |
|---------|---------|---------|
| `AI_EYES_MODEL_ID` | `google/siglip2-so400m-patch14-384` | HuggingFaceモデル |
| `AI_EYES_MODEL_REVISION` | 固定されたコミットSHA | モデルリビジョン。**40文字の16進数コミットSHAである必要があります。** `main`、タグ、または空の値は、フォールバックではなく、ハードなロードエラーになります。詳細は以下をご覧ください。 |
| `AI_EYES_MODEL_DIR` | HFデフォルトキャッシュ | モデルキャッシュディレクトリ |
| `AI_EYES_DEVICE` | 利用可能な場合は`cuda`、そうでない場合は`cpu` | torchデバイス。リテラルなデバイス（`cuda`、`cpu`、`cuda:1`）を設定します。`auto`の値はありません。`AI_EYES_DEVICE=auto`はエラーを発生させます。 |
| `AI_EYES_DEFAULT_THRESHOLD` | `0.02` | `image_contains`のデフォルト閾値 |
| `AI_EYES_LOG_LEVEL` | `WARNING` | ログの詳細度：`DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `AI_EYES_EAGER_LOAD` | 設定なし | 真の場合、起動時にモデルをロードし、壊れたモデル/キャッシュが最初のツール呼び出しではなく、すぐに失敗するようにします。 |
| `AI_EYES_DTYPE` | フル精度 | VRAMを半分にするには、`float16` / `bfloat16`を設定します。 |
| `AI_EYES_EMBED_CACHE` | `64` | インメモリの画像埋め込みメモサイズ。パス + mtime + サイズに基づいてキーが設定されるため、書き換えられたファイルは再測定され、古いデータは決して使用されません。ディスクやサイドカーファイルは使用しません。 |

### 再現性 - 重みは固定されています

スコアは、どの重みがそれを生成したかを知っている場合にのみ意味を持ちます。モデルリビジョンは**特定のコミットSHAに固定されており**、すべてのロード時に渡され、**数値を含むすべてのペイロードで報告されます。**数か月離れて実行された2つのインストールでは、同じ入力に対して同じスコアが返されます。

リビジョンを設定しない場合、デフォルト値が使用されます。浮動ブランチは決して使用されません。オーバーライドは、別の40文字のSHAとしてのみ有効です（オペレーターの意図）。`main`、タグ、または空の文字列は、明確なメッセージとともにロードを失敗させます。

**Logging:** The server logs under the `ai_eyes_mcp` logger to **stderr** (stdout is the MCP protocol channel). Set the level with `AI_EYES_LOG_LEVEL` (above), or attach your own handlers to `logging.getLogger("ai_eyes_mcp")`.

**最初の呼び出し：**モデルは遅延ロードされます。**最初の**画像ツール呼び出しでは、SigLIP2がダウンロード/ロードされます（GPU上では約10〜20秒、初回ダウンロードの場合はより時間がかかります）。その後の呼び出しは約100msです。サーバーの起動時にロードするには、`AI_EYES_EAGER_LOAD=1`を設定するか、`eyes_status`を呼び出して、ロードせずに`loaded`を報告します。**コールドサーバーでは無料ではありません** - 最初の呼び出しには、ライブラリのインポート（約10秒、ウォーム状態になると約2ms）という初期コストがかかります。

## スコアの仕組み

SigLIP2は**シグモイド**スコアリングを使用し、ソフトマックスは使用しません。各画像-テキストペアに対して独立した確率（0〜1）が割り当てられます。

あくまで概略的な説明です。これらの範囲はクエリ間や画像のスタイル間で転送されず、次のセクションでその理由を説明します。

- **高い** (>0.1)：強い視覚的マッチ
- **低い** (<0.01)：弱いまたは一致しない
- **中程度** (0.01〜0.1)：曖昧

スコアは相対的なものではありません。複数のクエリで、同じ画像に対して高いスコアが生成される可能性があります（例：剣と盾の両方を持つ画像）。

### ⚠ クエリの言い回しが重要です。堅牢な判断には、`image_classify`を使用することをお勧めします

SigLIP2シグモイドスコアは**クエリの言い回しに敏感です**：同じ画像の絶対スコアは、表現によって大きく変動します（スタイルが一致するフレーズは、一般的なフレーズよりも10〜100倍高いスコアになる可能性があります）。したがって、固定された`threshold`には、ユースケースごとにクエリを調整する必要があり、閾値は画像のスタイル間で転送されません。

さまざまな入力に対して、確実な肯定/否定の判断を行うには、**`image_classify`** を使用することをお勧めします。これは、候補ラベルを互いにランク付けし、絶対的なスコアの大きさに影響を受けません。クエリの表現と画像のスタイルを両方制御できる場合にのみ、調整された閾値を使用して `image_contains` を使用してください。`eyes_status` は、その `scoring_guidance` フィールドでこれに類似しています。

デフォルトの閾値（`0.02`）は、普遍的なカットオフではなく、寛容な下限です。クエリと画像のスタイルに合わせて調整するか、`image_classify` を使用してください。

## アーキテクチャ

```
engine.py          Standalone SigLIP2 wrapper — no MCP dependency.
                   Lazy-loads model on first inference call.
                   Importable directly for non-MCP use cases.

server.py          FastMCP wrapper that exposes engine methods as MCP tools.
                   Thin layer: input validation, error shaping, tool metadata.

__main__.py        Entry point for `python -m ai_eyes_mcp`.
```

`engine.py` はコアであり、モデルのロード、デバイスの選択、およびすべての推論ロジックを管理します。`server.py` は torch に直接アクセスすることはありません。すべてをエンジンに委譲します。つまり、`from ai_eyes_mcp.engine import SigLIPEngine` を使用して、FastMCP をインポートせずに、任意の Python スクリプトで使用できます。

ai-eyes-mcp は画像を評価するだけであり、それ以上のことは行いません。数値に対してどのような操作を行うかについては、関与しません（カタログ化、スプライト パイプライン、生成されたアセットに対する CI ゲートなど）。これは、消費者の責任です。

## ツールリファレンス

### `image_contains`

```
image_contains(image_path, query, threshold=0.02)
```

画像に、クエリで記述されたものが含まれているかどうかを確認します。独立したシグモイド スコア（0〜1）を返します。

| パラメータ | タイプ | 必須 | 説明 |
|-----------|------|----------|-------------|
| `image_path` | 文字列 | はい | 画像ファイルへの絶対パス |
| `query` | 文字列 | はい | 検索する内容（例：「剣を持っている人」） |
| `threshold` | 浮動小数点数 | no | 肯定の判断に対するスコア閾値（デフォルトは 0.02） |

戻り値：`{present, score, threshold, query, truncated, revision, elapsed_ms}`

`truncated: true` は、クエリがテキスト エンコーダーの 64 トークン制限を超えたことを意味し、**スコアは最初の 64 トークンのみを反映します**。数としてではなく、不完全なものとして扱ってください。

### `image_classify`

```
image_classify(image_path, labels)
```

複数の候補ラベルに対して画像を評価します。独立したシグモイド スコア（softmax ではありません）を返します。

| パラメータ | タイプ | 必須 | 説明 |
|-----------|------|----------|-------------|
| `image_path` | 文字列 | はい | 画像ファイルへの絶対パス |
| `labels` | 文字列[] | はい | スコアする候補ラベル（最大 20 個） |

戻り値：`{scores, best, best_score, truncated, revision, elapsed_ms}`

`best` は、フル精度スコアから選択されます。表示される値は、その選択と一貫性があるように丸められます。

### `image_compare`

```
image_compare(image_a, image_b, baselines=None)
```

SigLIP2 の埋め込みのコサイン類似度を使用して、2 つの画像間の視覚的な類似性を計算します。

| パラメータ | タイプ | 必須 | 説明 |
|-----------|------|----------|-------------|
| `image_a` | 文字列 | はい | 最初の画像への絶対パス |
| `image_b` | 文字列 | はい | 2 番目の画像への絶対パス |
| `baselines` | 文字列[][] | no | スタイルにおいて一致しない画像のペア。A-B は、この下限を超えた場合にのみ分離されたものとしてカウントされます。 |

**`baselines` がない場合、数値は測定値であり、判断ではありません** — ペイロードには `incomplete: true` が含まれます。類似度の下限は、画像スタイル間で転送されないため、ツールはそれを自動的に生成しません。スプライト スタイルの異なるキャラクターの 6 組に対して測定された結果：**0.698〜0.836** で、同じ画像同士の場合は 1 です。そこから固定のカットオフ値を設定すると、写真、スクリーンショット、レンダリングには不適切になるため、コントラストを明示的に指定する必要があります。これは、`image_verify` が代替案を受け取る方法とまったく同じです。

戻り値：`{similarity, separated, incomplete, margin, baseline_max, confidence, image_a, image_b, revision, elapsed_ms}`

### `image_score_batch`

```
image_score_batch(image_paths, query, threshold=0.02)
```

単一のクエリに対して複数の画像を評価します。1 回の呼び出しで最大 100 個の画像を使用できます。

| パラメータ | タイプ | 必須 | 説明 |
|-----------|------|----------|-------------|
| `image_paths` | 文字列[] | はい | 画像の絶対パスのリスト |
| `query` | 文字列 | はい | 検索する内容 |
| `threshold` | 浮動小数点数 | no | スコア閾値（デフォルトは 0.02） |

戻り値：`{query, threshold, total, scored, present, absent, errors, error_details?, results: [{path, score, present}], truncated, revision, elapsed_ms}`

### `image_verify`

```
image_verify(image_path, target, alternatives)
```

正直な**相対的な**判断 — `target` を、呼び出し元が提供した `alternatives` に対してランク付けします（必須、≥1）。決定 + マージン + 信頼度を返します。SigLIP のクエリの表現に対する感応性を考慮すると、これは相対的であるため、絶対的な閾値ではありません。生のスコアが必要な場合は `image_contains` を使用し、完全なランク付けを行うには `image_classify` を使用します。

| パラメータ | タイプ | 必須 | 説明 |
|-----------|------|----------|-------------|
| `image_path` | 文字列 | はい | 画像への絶対パス |
| `target` | 文字列 | はい | 検証する仮説 |
| `alternatives` | 文字列[] | はい | ランク付けする代替案（≥1） |

戻り値：`{present, target, target_score, best_alternative, best_alternative_score, margin, confidence, truncated, revision, elapsed_ms}` — `confidence` は `high` / `moderate` / `low — inconclusive` で、測定されたギャップを記述します。

### `image_rank`

```
image_rank(reference, candidates, k=5, baselines=None)
```

単一の参照に対して候補を類似度でランク付けします。参照は、候補ごとにエンコードするのではなく、**一度だけ**エンコードされます。

| パラメータ | タイプ | 必須 | 説明 |
|-----------|------|----------|-------------|
| `reference` | 文字列 | はい | 参照画像への絶対パス |
| `candidates` | 文字列[] | はい | ランク付けする候補画像のパス |
| `k` | 整数 | no | 返す最大一致数（デフォルトは 5） |
| `baselines` | 文字列[][] | no | スタイルにおいて一致しないペア — それ以下では何も「近い」とは見なされません。 |

戻り値：`{matches, nothing_close, incomplete, baseline_max, k, reference, revision, elapsed_ms}`

**`baselines` がない場合、ランク付けは測定値であり、「これらは一致する」というものではありません**（`incomplete: true`）。ベースラインを使用すると、呼び出し元が提供した下限を超える候補のみが一致するものとなり、いずれもそれを超えない場合は、`matches` は**空**になります。常に k 個の結果を返すランク付け動詞は、信号がない状況では確実な答えとなります。このツールはそのようなことはしません。

### `eyes_status`

```
eyes_status()
```

サーバーの状態を確認します。モデルのロードはトリガーされません。

戻り値：`{model_id, revision, device, dtype, loaded, cache_dir, parameters?, vram_mb?, scoring_guidance, note?}`

`loaded` が true の場合、`parameters`（`1136M`）と `vram_mb`（CUDA のみ）も返されます。

### `eyes_selftest`

```
eyes_selftest()
```

いくつかのバンドルされた参照画像でモデルを実行し、期待される順序が維持されていることを確認します。これにより、インストールが正しくロードされ、SigLIP2 が調整されていることが証明されます。まだロードされていない場合は、モデルをロードします。

戻り値：`{passed, checks: [{name, expected, measured_a, measured_b, ok}], model_id, device, torch_version, transformers_version}`

`measured_a` / `measured_b` はフル解像度で報告されます — ごくわずかなゼロ以外のスコアは、科学表記法（例：`4.3813e-08`）で表示され、`0` に縮小されることはありません。

## セキュリティと信頼性

このツールは**ローカルでのみ**動作します。

- **アクセスされるデータ:** ローカルの画像ファイル（読み取り専用）、HuggingFace モデル キャッシュ（一度ダウンロード）
- **ネットワーク:** すべての*推論*はローカルで行われ、外部への通信はありません。唯一の例外は、**初回実行時**に約 1.6 GB のモデル ウェイトが HuggingFace からダウンロードされることです。その後、このツールはネットワークにアクセスすることはありません。
- **エアギャップまたは外部通信監査されたデプロイの場合:** キャッシュを事前に設定し、`AI_EYES_MODEL_DIR` をその場所にポイントします。その後、ツールはいかなる時点でもネットワーク呼び出しを行いません。
- **機密情報の処理なし** — 資格情報や API キーは読み取らず、保存せず、送信しません。
- **テレメトリーなし** — 何も収集または送信されません。
- **ファイル変更なし** — 画像ファイルは読み取り専用で開かれ、変更されることはありません。
- **危険な操作なし** — 削除、強制終了、再起動などの操作はありません。
- **構造化されたエラーのみ** — スタック トレースはクライアントに公開されません。

## 要件

- Python 3.10 以降
- CUDA 対応の GPU を推奨。**デフォルト設定 `float32` で測定した VRAM は約 4.3 GB、`AI_EYES_DTYPE=float16` では約 2.2 GB です。** モデルは 1136M のパラメータを持ちます。「SO400M」という名前は、組み立てられたモデル全体ではなく、視覚処理部分を指します。
- CPU での実行も可能ですが、速度が遅くなります（約 10 倍）。
- 初回使用時に約 1.6 GB のモデルファイルをダウンロードします。

## 開発

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

**コンソールスクリプト** `pytest` を使用し、`python -m pytest` は使用しないでください。後者は作業ディレクトリを `sys.path` に設定するため、CI で検出される可能性のあるインポートエラーが隠れてしまうことがあります。

## ライセンス

MIT

---

[MCP Tool Shop](https://mcp-tool-shop.github.io/) によって作成されました。
