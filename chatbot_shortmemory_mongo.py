#
# MongoDB を用いたチャットボットのshortメモリ管理
#  - shortメモリ: 会話単位の履歴を保存し、次回以降の会話で利用可能にする
#  - long-termメモリ: ユーザープロファイルや過去の会話履歴を保存し、ユーザーの好みや傾向を学習する
# https://www.mongodb.com/ja-jp/docs/atlas/ai-integrations/langgraph/
# https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#add-short-term-memory

from langgraph.checkpoint.mongodb import MongoDBSaver
from typing import Annotated
from decode_checkpoints import DB_NAME
from llm_builder import build_azure_llm
from typing_extensions import TypedDict
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 修正: 正しい MongoDB 接続文字列
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "checkpointing_test_db"

def main():
    with MongoDBSaver.from_conn_string(conn_string=MONGO_URI, db_name=DB_NAME-) as checkpointer:
        def call_model(state: State):
            llm = build_azure_llm("gpt-4o")
            response = [llm.invoke(state["messages"])]
            return {"messages": response}
        
        builder = StateGraph(MessagesState)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "short_memory_example2"}}

        print("Mongo-backed short-memory REPL. Type 'quit' or 'exit' or Ctrl-C to end.")
        try:
            while True:
                try:
                    user_input = input("User: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                for chunk in graph.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config,
                    stream_mode="values",
                ):
                    chunk["messages"][-1].pretty_print()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

