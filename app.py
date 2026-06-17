from flask import Flask, json, jsonify, request,render_template
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
import base64
import mimetypes
app = Flask(__name__)

#region Load the database
# Load the database
with open('database.json', 'r') as f:
    db = json.load(f)


user_names = [user["name"] for user in db["users"]]
print(user_names)
#endregion
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

If the user refers to a previous image, previous field photo, previous parcel, or earlier agricultural context, classify it as AGRICULTURE_RELATED.
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

def image_to_base64(file):
    image_bytes = file.read()
    mime_type = file.content_type
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return mime_type, image_base64

area_ids = [
    area["area_id"]
    for user in db["users"]
    for area in user["saved_areas"]
]

# region prompt classification and agriculture agent
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

# endregion

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


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message")
    area_id = request.form.get("area_id")
    parcel_context = request.form.get("parcel_context")
    file = request.files.get("file")

    print("Message:", message)
    print("Area ID:", area_id)
    print("Parcel context:", parcel_context)

    response_text = ""

    if message and file:
        print("Uploaded file:", file)

        mime_type, image_base64 = image_to_base64(file)

        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": message
                },
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{image_base64}"
                }
            ]
        )

        for r in with_history.stream(
            {"messages": [human_message]},
            config=config,
        ):
            print(r.content, end="")
            response_text += r.content
        

    
    elif message and area_id:
        area_info = next(
            (
                area
                for user in db["users"]
                for area in user["saved_areas"]
                if area["area_id"] == area_id
            ),
            None
        )
        context_message_with_json = f"""
            Spatial context from mock JSON database:

            {area_info}

            Use this spatial context to answer questions about this user's fields, parcels,
            area_id, current crop, coordinates, NDVI, and NDWI.
            """
        # Gemini does not accept a SystemMessage inside the conversation
        # messages list (only the leading system instruction from the prompt
        # template). Merge the spatial context into the human message instead.
        human_message = HumanMessage(
            content=f"{context_message_with_json}\n\nUser question: {message}"
        )
        for r in with_history.stream(
                    {
                        "messages": [human_message]
                    },
                    config=config,
            ):
                print(r.content)
                response_text += r.content
    elif message: 
        for r in with_history.stream(
                    {
                        "messages": [HumanMessage(content=message)]
                    },
                    config=config,
            ):
                print(r.content)
                response_text += r.content


    return jsonify({
        "response": response_text
    })

if __name__ == "__main__":
    app.run(debug=True)
