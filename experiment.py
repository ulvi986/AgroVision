from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

store = {}

system_prompt_classification_agent = f"""
You are an expert classification assistant.

Your task is to determine whether a user's question is related to Agriculture, GIS (Geographic Information Systems), Remote Sensing, Precision Agriculture, Farming, Crops, Soil Science, Irrigation, Fertilizers, Plant Diseases, Weather Impact on Agriculture, Agricultural Technology, Land Management, Satellite Imagery Analysis, Environmental Monitoring, or any closely related agricultural domain.

Carefully analyze the user's query and classify it into one of the following categories:

- AGRICULTURE_RELATED
- NOT_AGRICULTURE_RELATED

Return only the classification label and nothing else.

Examples:

User: How can I increase wheat yield?
Output: AGRICULTURE_RELATED

User: What fertilizer is best for tomatoes?
Output: AGRICULTURE_RELATED

User: How do I analyze NDVI from Sentinel-2 imagery?
Output: AGRICULTURE_RELATED

User: What is Python programming?
Output: NOT_AGRICULTURE_RELATED

User: Solve this calculus problem.
Output: NOT_AGRICULTURE_RELATED

User: Explain object-oriented programming.
Output: NOT_AGRICULTURE_RELATED

If there is any uncertainty, classify based on the primary intent of the question. If the question is not directly related to agriculture, GIS, remote sensing, environmental monitoring, farming, crops, soil, irrigation, or agricultural technology, return NOT_AGRICULTURE_RELATED.
"""


def get_history(session_id : str) -> BaseChatMessageHistory:
    """
    Get the chat history for a given session ID.
    If no history exists, create a new one.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_classification_agent),
        MessagesPlaceholder(variable_name="history"),
    ]
)

chain = prompt | model
config = {"configurable": {"session_id": "firstChat"}}
with_message_history = RunnableWithMessageHistory(chain, get_history)

if __name__ == "__main__":
    while True:
        user_input = input("Please input your question:  ")
        for r in with_message_history.stream(
                {
                    "messages": [HumanMessage(content=user_input)]
                },
                config=config,
        ):
            print(r.content)