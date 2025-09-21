#
# langraphを用いた簡単なchatbot(memoryあり)
# https://langchain-ai.github.io/langgraph/tutorials/get-started/3-add-memory/#2-compile-the-graph#
#
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from llm_builder import build_azure_llm
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot_with_memory(state: State):
    llm = build_azure_llm("gpt-4o")
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot_with_memory", chatbot_with_memory)
graph_builder.add_edge(START, "chatbot_with_memory")
graph_builder.add_edge("chatbot_with_memory", END)
# チェックポイントを設定します。InMemorySaver はメモリ上に保存
memory_saver = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory_saver)
config = {"configurable": {"thread_id": "111111111111111111"}}

# -- 実行用ユーティリティ ---------------------------------------------
def stream_graph_updates(user_input: str):
    """
    ユーザー入力を受け取り、graph をストリーミング実行して
    途中の更新（ここでは LLM の応答）を順次出力します。
    途中の状態を保存するため、config に thread_id を含めています。
    """
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()


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
