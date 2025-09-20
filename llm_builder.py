from langchain_openai import AzureChatOpenAI

from llm_parameters import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT_LANGCHAIN,
    BASIC_ENCODE,
    SYSTEM_CODE,
)

MAX_RETRIES = 2


def build_azure_llm(deployment: str) -> AzureChatOpenAI:
    from pydantic import SecretStr

    if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT_LANGCHAIN and deployment):
        raise RuntimeError("Azure OpenAI settings missing.")
    return AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=SecretStr(AZURE_OPENAI_API_KEY),
        azure_endpoint=AZURE_OPENAI_ENDPOINT_LANGCHAIN,
        api_version=AZURE_OPENAI_API_VERSION,
        max_retries=MAX_RETRIES,
        default_headers={
            "system-code": str(SYSTEM_CODE),
            "Authorization": f"Basic {BASIC_ENCODE}",
        },
    )
