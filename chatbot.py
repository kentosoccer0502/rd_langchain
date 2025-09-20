#
# langraphを用いた簡単なchatbot(memoryなし)
# https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/#prerequisites
#
#

from typing import Annotated

from llm_builder import build_azure_llm
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# -- 設定 ---------------------------------------------------------------
# Azure やその他の LLM プロバイダでのデプロイ名（環境に合わせて変更）
DEPLOYMENT = "gpt-4o"


# -- State の定義（データの形） ----------------------------------------
# State はグラフが扱う「状態」を表します。
# ここでは messages（メッセージ履歴）だけを持つシンプルな形にしています。
class State(TypedDict):
    # messages は辞書のリスト（role / content）で構成されます。
    # add_messages リデューサーを使うことで既存の履歴に追加していきます。
    messages: Annotated[list, add_messages]


# StateGraph を作成します。これがノード（関数）とエッジ（遷移）を持つ
# 『ステートマシン』の役割を果たします。
graph_builder = StateGraph(State)


# -- LLM の準備 -------------------------------------------------------
# 実際に応答を作るためのモデルを初期化します。llm_builder.build_azure_llm
# は環境変数などを読み込んで Azure のチャットモデルを作るラッパー関数と想定。
llm = build_azure_llm(DEPLOYMENT)


# -- ノード（chatbot）の定義 ------------------------------------------
def chatbot(state: State):
    """
    ノード関数の例。
    - 入力: 現在の state（ここでは state["messages"] に会話履歴が入る）
    - 処理: LLM を呼び出して応答を作る
    - 出力: messages に追加するための辞書を返す

    LangGraph のノード関数は常に『State を受け取り、State の更新を辞書で返す』
    ことを期待されます。
    """
    return {"messages": [llm.invoke(state["messages"]) ]}


# ノードをグラフに登録します。
# 第一引数: ノード名（ユニーク）
# 第二引数: 実行する関数やオブジェクト
graph_builder.add_node("chatbot", chatbot)

# 開始（START）から chatbot を実行し、chatbot の後に終了（END）するシンプルなフロー
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# グラフをコンパイルして実行可能なオブジェクトを作成
graph = graph_builder.compile()


# -- 実行用ユーティリティ ---------------------------------------------
def stream_graph_updates(user_input: str):
    """
    ユーザー入力を受け取り、graph をストリーミング実行して
    途中の更新（ここでは LLM の応答）を順次出力します。

    - graph.stream(..., stream_mode="updates") はイベントを逐次受け取るための仕組みです。
    - event の中身を見て最新のメッセージ（value["messages"][-1]）を表示しています。
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode="updates"):
        for value in event.values():
            # 最新のメッセージの content を出力
            print("Assistant:", value["messages"][-1].content)


# -- メインループ（対話） ---------------------------------------------
# シンプルな REPL。ユーザーが入力すると graph を実行して応答を表示します。
while True:
    try:
        user_input = input("User: ")
        # 終了コマンド
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception:
        # 入力が使えない環境（例えば一部のノート環境）へのフォールバック
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
