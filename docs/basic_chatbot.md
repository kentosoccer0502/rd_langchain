Source: https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/

# Build a basic chatbot

In this tutorial, you will build a basic chatbot. This chatbot is the basis for the following series of tutorials where you will progressively add more sophisticated capabilities, and be introduced to key LangGraph concepts along the way. Let's dive in! 🌟
Source: https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/

# 基本的なチャットボットを作る（初心者向けに詳しく説明）

このドキュメントでは、LangGraph を使って「何をしているかが分かるように」ステップごとに説明しながら、基本的なチャットボットを作る方法を解説します。

最初に「全体像」と「実行の契約（contract）」を示します。コードに入る前にここを読むと、各ステップが何のためにあるのか理解しやすくなります。

## 全体像（ざっくり）

- 入力: ユーザーのメッセージ（例: "こんにちは"）
- 出力: LLM（チャットモデル）が生成した応答を messages に追加した状態
- 流れ: グラフ（StateGraph）を定義 → ノード（LLM 呼び出し）を追加 → 開始点/終了点をつなぐ → コンパイル → 実行

このチュートリアルで作るものは「ユーザーの入力を受け取り、LLM に投げて応答を返す」非常にシンプルなボットです。LangGraph はこれを「状態（State）」と「ノード（関数）」、そしてノード間の「遷移（エッジ）」で表現します。

## 実行の契約（Inputs / Outputs / 成功基準）

- 入力（Input）: State（ここでは messages というリストを含む辞書）
- 出力（Output）: State の更新（メッセージが追加された辞書）
- 成功: graph を実行したら、messages に assistant の応答が追加されること

※「契約」は短くまとめることで、ノードやテストを作るときに役立ちます。

## データの形（State と messages の中身）

このチュートリアルでは State が次のような形を取ります（Python の型注釈で表現）:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

`messages` は通常、次のような辞書のリストです。

例:

```python
[{"role": "user", "content": "こんにちは"},
 {"role": "assistant", "content": "こんにちは！何をお手伝いしましょうか？"}]
```

ここで重要なのは「messages は履歴（リスト）である」という点です。`add_messages` というリデューサーを使うことで、新しい応答を既存の履歴に追加できます（上書きしない）。

## ステップ概要（何をするか・なぜ必要か）

1) パッケージをインストールする
   - LangGraph と LangSmith（トレース確認）はインストールしておきます。

```bash
pip install -U langgraph langsmith
```

2) StateGraph を作る（状態と更新ルールを定義）
   - State（データの形）を定義し、StateGraph を初期化します。

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
```

3) チャットモデル（LLM）を選ぶ
   - OpenAI、Anthropic、Google などプロバイダを選んで初期化します（環境変数で API キーを設定）。

4) ノード（chatbot 関数）を追加する
   - ノードは現在の State を受け取り、State の更新（辞書）を返す普通の関数です。

```python
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"]) ]}

graph_builder.add_node("chatbot", chatbot)
```

5) エントリ（START）と終了（END）をつなぐ
   - どこから処理を始め、どこで終えるかを明示します。

```python
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```

6) コンパイルして実行可能なグラフを作る

```python
graph = graph_builder.compile()
```

7) （任意）可視化して構造を確認する

8) 実行して会話を試す
   - ユーザー入力を State に入れて graph を流し、出力メッセージを表示します。

## よくあるトラブルと対処（初心者向け）

- API キーが見つからない / 認証エラー
  - 環境変数（例: OPENAI_API_KEY）を設定しているか確認してください。

- 依存パッケージが足りない
  - 指定の extras（例: langchain[openai]）や langgraph の依存をインストールしてください。

- messages に期待したフォーマットが入っていない
  - `messages` は辞書のリスト（role / content の形）であることを確認。

- LLM の呼び出し方法が変わった（API 仕様差分）
  - `init_chat_model` や `llm.invoke()` の挙動はバージョンによって異なることがあるため、公式ドキュメントを参照してください。

---

下にサンプルコード（元の内容）をそのまま載せます。必要ならこのコードをローカルにコピーして実行してください。

```python
from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"]) ]}


# ノード登録と接続
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
```

実行例や可視化、次のチュートリアル（検索ツールの追加）については元のチュートリアルの流れに従ってください。

## 次のステップ

このボットは学習データに基づく知識のみを持っているため、次はウェブ検索などのツールを追加して知識を補強する流れになります。[次へ → 2-add-tools.md](./2-add-tools.md)

## 参考
- [LangGraph 公式ドキュメント](https://langchain-ai.github.io/langgraph/)
- [Youtube 動画(結構わかりやすい)](https://www.youtube.com/watch?v=CvqQFqRLjeQ)