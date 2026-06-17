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
from langchain_core.runnables import RunnableLambda # for function calling
from langchain_core.messages import AIMessage, SystemMessage
import json

with open('database.json', 'r') as file:
    data = json.load(file)

user_names = [user["name"] for user in data["users"]]
print(user_names)

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

# region storing
store = {}
# endregion

# region system prompt for classification agent
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
#endregion


#region system prompt for agriculture agent
system_prompt_agriculture_agent = f""" 
You are an expert agriculture assistant.
Your task is to provide accurate and helpful information related to agriculture, GIS (Geographic Information Systems), Remote Sensing, Precision Agriculture, Farming, Crops, Soil Science, Irrigation, Fertilizers, Plant Diseases, Weather Impact on Agriculture, Agricultural Technology, Land Management, Satellite Imagery Analysis, Environmental Monitoring, or any closely related agricultural domain.
"""
# endregion




def get_history(session_id : str) -> BaseChatMessageHistory:
    """
    Get the chat history for a given session ID.
    If no history exists, create a new one.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def find_user_by_name(user_input, database_path="database.json"):
    with open(database_path, "r") as f:
        data = json.load(f)

    for user in data["users"]:
        if user["name"].lower() in user_input.lower():
            return user
    


prompt_classification_agent = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_classification_agent),
        MessagesPlaceholder(variable_name="messages"),

    ]
)

prompt_agriculture_agent = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_agriculture_agent),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages"),

    ]
)

classification_chain = prompt_classification_agent | model
agriculture_chain = prompt_agriculture_agent | model


def router(inputs):
    result = classification_chain.invoke(
        {
            "messages": inputs["messages"]
        }
    )

    label = result.content.strip()

    if label == "AGRICULTURE_RELATED":
        return agriculture_chain.invoke(inputs)

    return AIMessage(
        content="I am an AI specialized exclusively in agriculture, GIS, and related fields, so I cannot answer this question."
    )



router_chain = RunnableLambda(router)


config = {"configurable": {"session_id": "first"}}
with_history = RunnableWithMessageHistory(
    router_chain,
    get_history,
    input_messages_key="messages",
    history_messages_key="history"
)


if __name__ == "__main__":
    
    while True:
        user_input = input("Please input your question:  ")
        user = find_user_by_name(user_input)
        if user:
            context_message_with_json = f"""
            Spatial context from mock JSON database:

            User ID: {user["user_id"]}
            User Name: {user["name"]}

            Saved Areas:
            {user["saved_areas"]}

            Use this spatial context to answer questions about this user's fields, parcels,
            area_id, current crop, coordinates, NDVI, and NDWI.
            """
            for r in with_history.stream(
                    {
                        "messages": [
                            SystemMessage(content=context_message_with_json),
                            HumanMessage(content=user_input)
                                     ]
                    },
                    config=config,
            ):
                print(r.content)


        else:
            for r in with_history.stream(
                    {
                        "messages": [HumanMessage(content=user_input)]
                    },
                    config=config,
            ):
                print(r.content)
        