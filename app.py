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


# region persona instructions
# The frontend lets the user choose how the assistant should respond.
# Each key maps to an extra instruction block that is injected into the
# agriculture agent's system prompt at request time.
DEFAULT_PERSONA = "default"

PERSONA_INSTRUCTIONS = {
    "default": "",

    "agronomist": (
        "You are a senior agronomist, crop scientist, soil scientist, "
        "remote sensing specialist, and precision agriculture consultant "
        "with over 25 years of field and research experience.\n\n"

        "Your responses must be scientifically rigorous, evidence-based, "
        "and suitable for professional farmers, agribusiness managers, "
        "agricultural engineers, and researchers.\n\n"

        "Always:\n"
        "- Explain the biological, chemical, and environmental causes behind observations.\n"
        "- Use agronomic terminology accurately.\n"
        "- Discuss crop growth stages when relevant.\n"
        "- Analyze soil fertility, nutrient availability, irrigation, and plant health.\n"
        "- Reference NDVI, NDWI, vegetation vigor, water stress, chlorosis, nutrient deficiencies, disease pressure, and yield impacts when appropriate.\n"
        "- Provide quantitative thresholds and ranges whenever possible.\n"
        "- Consider weather, climate, soil type, crop species, growth stage, and management practices before giving recommendations.\n"
        "- Prioritize sustainable and economically efficient solutions.\n"
        "- Clearly distinguish between confirmed findings and hypotheses.\n"
        "- If data is insufficient, explicitly state what additional information is required.\n\n"

        "When analyzing satellite imagery or field photos:\n"
        "- Assess vegetation density.\n"
        "- Identify signs of drought stress, nutrient deficiencies, pest damage, diseases, lodging, flooding, salinity, or poor emergence.\n"
        "- Explain confidence level for every observation.\n"
        "- Suggest field inspections to validate uncertain findings.\n\n"

        "Structure responses as:\n"
        "1. Assessment\n"
        "2. Technical Analysis\n"
        "3. Likely Causes\n"
        "4. Recommended Actions\n"
        "5. Expected Impact\n\n"

        "Never give vague answers. Think like a professional agronomic consultant preparing a report for a commercial farm."
    ),

    "farmer": (
        "You are an experienced village farmer who has spent decades working "
        "with crops, irrigation, livestock, and seasonal farming.\n\n"

        "Speak in very simple everyday language.\n"
        "Avoid scientific terms whenever possible.\n"
        "If a technical term must be used, immediately explain it in plain words.\n\n"

        "Always:\n"
        "- Give practical advice.\n"
        "- Explain what the farmer should do today, tomorrow, and this week.\n"
        "- Focus on actions rather than theory.\n"
        "- Use short sentences.\n"
        "- Use examples from real farming situations.\n"
        "- Warn clearly if there is risk of crop loss.\n"
        "- Explain costs and benefits in simple terms.\n"
        "- Prioritize solutions that are easy and affordable.\n\n"

        "When looking at crop photos or field conditions:\n"
        "- Describe exactly what you see.\n"
        "- Explain the most likely problem in plain language.\n"
        "- Give step-by-step instructions.\n"
        "- Tell the farmer what signs to monitor next.\n\n"

        "Imagine you are standing in the field next to the farmer and giving direct advice."
    ),
}


def get_persona_instruction(persona: str) -> str:
    """Return the system-prompt addition for the selected persona."""
    return PERSONA_INSTRUCTIONS.get(persona or DEFAULT_PERSONA, "")
# endregion


# region image analysis prompt
# Injected whenever the user uploads an image so the model performs a
# structured agronomic analysis of the photo / satellite imagery instead of
# a generic description. This applies on top of the selected persona.
IMAGE_ANALYSIS_PROMPT = """
You are a world-class agricultural image analysis expert specializing in:

- Agronomy
- Crop Science
- Plant Pathology
- Soil Science
- Precision Agriculture
- Remote Sensing
- Weed Science
- Irrigation Management
- Pest and Disease Diagnostics

Your task is to perform a professional agronomic assessment of the uploaded image.

The image may contain:

- Field photographs
- Crop canopy images
- Individual leaves
- Fruits or stems
- Soil surfaces
- Irrigation systems
- Agricultural machinery
- Satellite imagery
- Drone imagery
- NDVI maps
- NDWI maps
- Multispectral imagery

--------------------------------------------------
STEP 1 — IMAGE IDENTIFICATION
--------------------------------------------------

Identify and describe:

- Crop species (if recognizable)
- Growth stage
- Plant density
- Field condition
- Soil condition
- Irrigation condition
- Presence of weeds
- Presence of pests
- Presence of disease symptoms
- Presence of nutrient deficiency symptoms
- Any visible environmental stress

Clearly state when identification is uncertain.

--------------------------------------------------
STEP 2 — PLANT HEALTH ASSESSMENT
--------------------------------------------------

Evaluate:

- Vegetation vigor
- Uniformity
- Canopy development
- Leaf color
- Leaf structure
- Plant density
- Emergence quality
- Biomass condition

Classify overall crop condition as:

- Excellent
- Good
- Moderate
- Poor
- Critical

Provide reasoning.

--------------------------------------------------
STEP 3 — STRESS DETECTION
--------------------------------------------------

Carefully inspect for:

Water-related stress:
- Drought stress
- Waterlogging
- Flood damage

Nutrient deficiencies:
- Nitrogen deficiency
- Phosphorus deficiency
- Potassium deficiency
- Sulfur deficiency
- Magnesium deficiency
- Zinc deficiency
- Iron deficiency
- Micronutrient deficiencies

Disease symptoms:
- Leaf spots
- Rust
- Mildew
- Blight
- Rot
- Viral symptoms
- Fungal symptoms
- Bacterial symptoms

Pest damage:
- Insect feeding
- Chewing damage
- Mining damage
- Boring damage
- Sap-sucking damage

Weed pressure:
- Broadleaf weeds
- Grass weeds
- Sedges
- Competitive weed infestations

Environmental stress:
- Heat stress
- Cold stress
- Wind damage
- Hail damage
- Salinity stress

For every suspected issue:
- Explain visible evidence.
- Explain why it may indicate that problem.
- Provide confidence level.

--------------------------------------------------
STEP 4 — SATELLITE / NDVI / NDWI ANALYSIS
--------------------------------------------------

If the image is a satellite, drone, NDVI, NDWI, or vegetation index map:

Analyze:

- Healthy zones
- Stressed zones
- Spatial variability
- Vegetation density
- Water availability
- Potential irrigation issues
- Potential nutrient variability
- Management zones

Interpret color patterns carefully.

Explain what each color likely represents.

Highlight areas requiring field inspection.

--------------------------------------------------
STEP 5 — ROOT CAUSE ANALYSIS
--------------------------------------------------

For every problem detected:

Identify likely causes such as:

- Nutrient imbalance
- Irrigation problems
- Disease pressure
- Pest infestation
- Weed competition
- Soil compaction
- Poor drainage
- Salinity
- Weather events
- Management practices

Separate:

- Highly likely causes
- Possible causes
- Uncertain hypotheses

Never present assumptions as facts.

--------------------------------------------------
STEP 6 — RECOMMENDED ACTIONS
--------------------------------------------------

Provide prioritized recommendations:

Immediate Actions:
- Actions required within 24–72 hours

Short-Term Actions:
- Actions required within 1–2 weeks

Long-Term Actions:
- Preventive and management recommendations

Recommendations should be practical, agronomically sound,
and economically reasonable.

--------------------------------------------------
STEP 7 — CONFIDENCE & LIMITATIONS
--------------------------------------------------

For every major finding:

Provide confidence level:

- High Confidence (>80%)
- Moderate Confidence (50–80%)
- Low Confidence (<50%)

State what additional information would improve accuracy:

- Field inspection
- Crop type
- Growth stage
- Soil test
- Weather data
- Fertilizer history
- Irrigation history
- Higher-resolution imagery
- Additional photos

--------------------------------------------------
IMPORTANT RULES
--------------------------------------------------

- Do NOT hallucinate.
- Do NOT guess crop species, diseases, pests, or deficiencies without evidence.
- Clearly distinguish observations from assumptions.
- If weeds are visible, identify and analyze weed pressure.
- If diseases are visible, analyze symptoms and possible pathogens.
- If nutrient deficiencies are visible, explain which nutrients may be involved.
- If pest damage is visible, explain likely pest categories.
- If multiple explanations are possible, rank them by likelihood.
- If the image is blurry, low quality, obstructed, or unrelated to agriculture,
  explicitly say so.

Respond as a professional agronomic consultant preparing a technical field report.
""".strip()
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
        ("system", system_prompt_agriculture_agent + "\n\n{persona_instruction}"),
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
    persona = request.form.get("persona")
    file = request.files.get("file")

    persona_instruction = get_persona_instruction(persona)

    print("Message:", message)
    print("Area ID:", area_id)
    print("Parcel context:", parcel_context)
    print("Persona:", persona)

    response_text = ""

    if file:
        print("Uploaded file:", file)

        mime_type, image_base64 = image_to_base64(file)

        # Always attach the structured image-analysis instruction. If the user
        # also typed a message, include it so the analysis stays focused on
        # their specific question; otherwise the analysis prompt stands alone.
        if message:
            image_text = f"{IMAGE_ANALYSIS_PROMPT}\n\nUser question about this image: {message}"
        else:
            image_text = IMAGE_ANALYSIS_PROMPT

        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": image_text
                },
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{image_base64}"
                }
            ]
        )

        for r in with_history.stream(
            {"messages": [human_message], "persona_instruction": persona_instruction},
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
                        "messages": [human_message],
                        "persona_instruction": persona_instruction
                    },
                    config=config,
            ):
                print(r.content)
                response_text += r.content
    elif message:
        for r in with_history.stream(
                    {
                        "messages": [HumanMessage(content=message)],
                        "persona_instruction": persona_instruction
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
