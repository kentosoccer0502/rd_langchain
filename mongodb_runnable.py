#
# MongoDB Chat Message History with Runnable chain
# https://python.langchain.com/docs/integrations/memory/mongodb_chat_message_history/
#
#
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from llm_builder import build_azure_llm
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

llm = build_azure_llm(deployment="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string="mongodb://localhost:27017",
        database_name="test_mongodbchatmessagehistory",
        collection_name="chat_histories_runnable",
    ),
    input_messages_key="question",
    history_messages_key="history",
)

# This is where we configure the session id
config = {"configurable": {"session_id": "test_session_id_1"}}

chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)

chain_with_history.invoke({"question": "Whats my name"}, config=config)


"""
MongoDB に保存された履歴の例
[{
  "_id": {
    "$oid": "68d1ed4a631f66993b662da9"
  },
  "SessionId": "test_session_id_1",
  "History": "{\"type\": \"human\", \"data\": {\"content\": \"Hi! I'm bob\", \"additional_kwargs\": {}, \"response_metadata\": {}, \"type\": \"human\", \"name\": null, \"id\": null, \"example\": false}}"
},
{
  "_id": {
    "$oid": "68d1ed4a631f66993b662daa"
  },
  "SessionId": "test_session_id_1",
  "History": "{\"type\": \"ai\", \"data\": {\"content\": \"Hi Bob! \\ud83d\\udc4b How can I assist you today? \\ud83d\\ude0a\", \"additional_kwargs\": {\"refusal\": null}, \"response_metadata\": {\"token_usage\": {\"completion_tokens\": 14, \"prompt_tokens\": 21, \"total_tokens\": 35, \"completion_tokens_details\": {\"accepted_prediction_tokens\": 0, \"audio_tokens\": 0, \"reasoning_tokens\": 0, \"rejected_prediction_tokens\": 0}, \"prompt_tokens_details\": {\"audio_tokens\": 0, \"cached_tokens\": 0}}, \"model_name\": \"gpt-4o-2024-11-20\", \"system_fingerprint\": \"fp_ee1d74bde0\", \"id\": \"chatcmpl-CIlQXdutYVsOw2OygoVJoo28JRiHH\", \"service_tier\": null, \"prompt_filter_results\": [{\"prompt_index\": 0, \"content_filter_results\": {\"hate\": {\"filtered\": false, \"severity\": \"safe\"}, \"self_harm\": {\"filtered\": false, \"severity\": \"safe\"}, \"sexual\": {\"filtered\": false, \"severity\": \"safe\"}, \"violence\": {\"filtered\": false, \"severity\": \"safe\"}}}], \"finish_reason\": \"stop\", \"logprobs\": null, \"content_filter_results\": {\"hate\": {\"filtered\": false, \"severity\": \"safe\"}, \"self_harm\": {\"filtered\": false, \"severity\": \"safe\"}, \"sexual\": {\"filtered\": false, \"severity\": \"safe\"}, \"violence\": {\"filtered\": false, \"severity\": \"safe\"}}}, \"type\": \"ai\", \"name\": null, \"id\": \"run--94cdddad-f770-4b95-bcd7-e8b4fb7cdac9-0\", \"example\": false, \"tool_calls\": [], \"invalid_tool_calls\": [], \"usage_metadata\": {\"input_tokens\": 21, \"output_tokens\": 14, \"total_tokens\": 35, \"input_token_details\": {\"audio\": 0, \"cache_read\": 0}, \"output_token_details\": {\"audio\": 0, \"reasoning\": 0}}}}"
},
{
  "_id": {
    "$oid": "68d1ed4b631f66993b662dac"
  },
  "SessionId": "test_session_id_1",
  "History": "{\"type\": \"human\", \"data\": {\"content\": \"Whats my name\", \"additional_kwargs\": {}, \"response_metadata\": {}, \"type\": \"human\", \"name\": null, \"id\": null, \"example\": false}}"
},
{
  "_id": {
    "$oid": "68d1ed4b631f66993b662dad"
  },
  "SessionId": "test_session_id_1",
  "History": "{\"type\": \"ai\", \"data\": {\"content\": \"Your name is Bob! \\ud83d\\ude0a How can I help you, Bob?\", \"additional_kwargs\": {\"refusal\": null}, \"response_metadata\": {\"token_usage\": {\"completion_tokens\": 15, \"prompt_tokens\": 45, \"total_tokens\": 60, \"completion_tokens_details\": {\"accepted_prediction_tokens\": 0, \"audio_tokens\": 0, \"reasoning_tokens\": 0, \"rejected_prediction_tokens\": 0}, \"prompt_tokens_details\": {\"audio_tokens\": 0, \"cached_tokens\": 0}}, \"model_name\": \"gpt-4o-2024-11-20\", \"system_fingerprint\": \"fp_ee1d74bde0\", \"id\": \"chatcmpl-CIlQYUMvHDkym3hPuG12QDkJKJsfd\", \"service_tier\": null, \"prompt_filter_results\": [{\"prompt_index\": 0, \"content_filter_results\": {\"hate\": {\"filtered\": false, \"severity\": \"safe\"}, \"self_harm\": {\"filtered\": false, \"severity\": \"safe\"}, \"sexual\": {\"filtered\": false, \"severity\": \"safe\"}, \"violence\": {\"filtered\": false, \"severity\": \"safe\"}}}], \"finish_reason\": \"stop\", \"logprobs\": null, \"content_filter_results\": {\"hate\": {\"filtered\": false, \"severity\": \"safe\"}, \"self_harm\": {\"filtered\": false, \"severity\": \"safe\"}, \"sexual\": {\"filtered\": false, \"severity\": \"safe\"}, \"violence\": {\"filtered\": false, \"severity\": \"safe\"}}}, \"type\": \"ai\", \"name\": null, \"id\": \"run--10780cce-3707-4176-a2e5-ee5a7977e50c-0\", \"example\": false, \"tool_calls\": [], \"invalid_tool_calls\": [], \"usage_metadata\": {\"input_tokens\": 45, \"output_tokens\": 15, \"total_tokens\": 60, \"input_token_details\": {\"audio\": 0, \"cache_read\": 0}, \"output_token_details\": {\"audio\": 0, \"reasoning\": 0}}}}"
}]
"""