#
# MongoDB Chat Message History
# https://python.langchain.com/docs/integrations/memory/mongodb_chat_message_history/
#
#
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

chat_message_history = MongoDBChatMessageHistory(
    session_id="test_session",
    connection_string="mongodb://localhost:27017",
    database_name="test_mongodbchatmessagehistory",
    collection_name="chat_histories",
)

chat_message_history.add_user_message("Hello")
chat_message_history.add_ai_message("Hi")

"""
# MongoDB に保存された履歴の例
{
  "_id": {
    "$oid": "68d1eb6174598cef24d5b03e"
  },
  "SessionId": "test_session",
  "History": "{\"type\": \"human\", \"data\": {\"content\": \"Hello\", \"additional_kwargs\": {}, \"response_metadata\": {}, \"type\": \"human\", \"name\": null, \"id\": null, \"example\": false}}"
}

{
  "_id": {
    "$oid": "68d1eb6174598cef24d5b03f"
  },
  "SessionId": "test_session",
  "History": "{\"type\": \"ai\", \"data\": {\"content\": \"Hi\", \"additional_kwargs\": {}, \"response_metadata\": {}, \"type\": \"ai\", \"name\": null, \"id\": null, \"example\": false, \"tool_calls\": [], \"invalid_tool_calls\": [], \"usage_metadata\": null}}"
}
"""

print(chat_message_history.messages)



