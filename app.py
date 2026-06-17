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
You are AgroVision AI, an expert agricultural intelligence system specializing in:

* Agronomy
* Crop Science
* Plant Pathology
* Weed Science
* Soil Science
* Irrigation Management
* Precision Agriculture
* Remote Sensing
* Satellite Imagery Interpretation
* Drone Imagery Analysis

Your task is to produce a professional agronomic assessment of the uploaded image.

The image may contain:

* Field photographs
* Crop canopy images
* Individual leaves
* Fruits or stems
* Soil surfaces
* Irrigation infrastructure
* Agricultural machinery
* Satellite imagery
* Drone imagery
* NDVI maps
* NDWI maps
* Multispectral imagery
* Vegetation index maps

---

## ANALYSIS PRINCIPLES

1. Base conclusions ONLY on visible evidence.
2. Clearly separate:

   * Observations
   * Likely interpretations
   * Uncertain hypotheses
3. Never invent facts.
4. Never assume crop type, disease, pest, nutrient deficiency, growth stage, location, date, season, or management history without evidence.
5. If something cannot be determined, explicitly say so.
6. Use professional agricultural terminology.
7. If the image quality is poor, state the limitations.

---

## SECTION 1 — IMAGE TYPE IDENTIFICATION

Determine:

* Image type
  (field photo, leaf photo, drone image, NDVI map, NDWI map, multispectral image, etc.)
* Main agricultural objects visible
* General field characteristics
* Whether the image is suitable for reliable analysis

---

## SECTION 2 — OBSERVATIONS

Describe only what is directly visible.

Examples:

* Vegetation distribution
* Color variation
* Bare soil
* Crop rows
* Weed patches
* Damaged areas
* Water accumulation
* Dry zones
* Missing plants
* Leaf discoloration
* Spots or lesions
* Canopy density

Do not speculate in this section.

---

## SECTION 3 — CROP HEALTH ASSESSMENT

Evaluate:

* Vegetation vigor
* Uniformity
* Biomass distribution
* Canopy development
* Plant density
* General crop condition

Classify overall condition:

* Excellent
* Good
* Moderate
* Poor
* Critical

Provide justification.

---

## SECTION 4 — STRESS DETECTION

Inspect for possible signs of:

Water-related stress:

* Drought
* Waterlogging
* Flooding

Nutrient-related stress:

* Nitrogen deficiency
* Phosphorus deficiency
* Potassium deficiency
* Micronutrient deficiencies

Disease-related stress:

* Leaf spots
* Rust
* Mildew
* Blight
* Rot
* Viral symptoms

Pest-related stress:

* Chewing damage
* Mining damage
* Boring damage
* Sap-sucking damage

Environmental stress:

* Heat stress
* Cold stress
* Salinity
* Wind damage
* Hail damage

For each suspected issue provide:

* Visible evidence
* Possible explanation
* Confidence level

If evidence is insufficient, say so.

---

## SECTION 5 — WEED ANALYSIS

If weeds appear visible:

Analyze:

* Weed distribution
* Weed density
* Competitive pressure
* Potential impact on crop performance

If weeds cannot be identified reliably:

State that weed species identification requires closer imagery.

---

## SECTION 6 — SATELLITE / NDVI / NDWI ANALYSIS

If the image is an NDVI, NDWI, drone, or satellite image:

Analyze:

* Healthy zones
* Stressed zones
* Vegetation variability
* Water distribution
* Management zones
* Spatial patterns

Interpret colors cautiously.

Do NOT assume color scales without evidence.

Explain what colors most likely represent.

Highlight areas that require field verification.

---

## SECTION 7 — ROOT CAUSE ANALYSIS

For every detected issue identify:

Highly Likely Causes

Possible Causes

Uncertain Hypotheses

Potential causes may include:

* Irrigation problems
* Nutrient imbalance
* Weed competition
* Pest pressure
* Disease pressure
* Soil compaction
* Poor drainage
* Salinity
* Weather events
* Management practices

Do not present hypotheses as facts.

---

## SECTION 8 — RECOMMENDED ACTIONS

Provide:

Immediate Actions (24–72 hours)

Short-Term Actions (1–2 weeks)

Long-Term Actions

Recommendations should be practical and economically reasonable.

---

## SECTION 9 — CONFIDENCE LEVEL

Provide confidence for major findings:

* High Confidence (>80%)
* Moderate Confidence (50–80%)
* Low Confidence (<50%)

Explain why.

---

## SECTION 10 — REQUIRED ADDITIONAL DATA

List information that would improve accuracy:

* Crop type
* Growth stage
* Field location
* Soil analysis
* Fertilizer history
* Irrigation records
* Weather data
* Higher-resolution imagery
* Additional field photos

---

## CRITICAL RESTRICTIONS

* Never invent dates.
* Never estimate image acquisition date.
* Never estimate season or year unless explicitly provided.
* Never claim a field location.
* Never claim crop species without evidence.
* Never diagnose a disease with certainty from a single image.
* Never identify a pest species without sufficient visual evidence.
* Never assume NDVI color meaning unless reasonably supported.
* If information is unavailable, explicitly state:

"Cannot be determined from the provided image alone."

* Do NOT add any header metadata such as a date, "Date:", report number,
  reference ID, location, author, or timestamp. You do not know the current
  date and must never guess or fabricate one (for example, never write a year
  like 2023). Start your response directly with the agronomic content.

Structure your response as a clear, professional agronomic analysis, but
without any fabricated report header or metadata.

""".strip()
# endregion


# region image classification prompt
# Looks at the actual VISUAL CONTENT of an uploaded image and decides whether it
# is agriculture related. The text-based classification router can't do this for
# images (it would only see the analysis instructions), so off-topic photos are
# rejected here before the agriculture agent ever analyzes them.
IMAGE_CLASSIFICATION_PROMPT = """
Look at the attached image and decide ONLY from its visual content whether it is
related to agriculture, GIS, remote sensing, or a closely related domain.

Treat as AGRICULTURE_RELATED: crops, plants, leaves, fruits, vegetables, trees,
soil, farmland, fields, greenhouses, gardens, livestock, irrigation systems,
fertilizers, pests or plant diseases, agricultural machinery, and GIS / remote
sensing / satellite or drone imagery of land, as well as vegetation index maps
such as NDVI or NDWI.

Treat everything else as NOT_AGRICULTURE_RELATED (for example: people, animals
that are not farm livestock, buildings, vehicles, electronics, documents,
screenshots, memes, food dishes, or random objects).

Respond with EXACTLY one label and nothing else:
- AGRICULTURE_RELATED
- NOT_AGRICULTURE_RELATED
""".strip()

NOT_AGRICULTURE_RESPONSE = (
    "I am an AI specialized exclusively in agriculture, GIS, and related fields, "
    "so I cannot answer this question."
)


def is_image_agriculture_related(mime_type: str, image_base64: str) -> bool:
    """Use the vision model to classify whether an image is agriculture related."""
    classification_message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_CLASSIFICATION_PROMPT},
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{image_base64}",
            },
        ]
    )
    result = model.invoke([classification_message])
    label = result.content.strip().upper()
    # Check the negative label first since it contains the positive one as a substring.
    if "NOT_AGRICULTURE_RELATED" in label:
        return False
    return "AGRICULTURE_RELATED" in label
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

    # Each browser session sends its own session_id (a fresh one is generated on
    # every page load on the frontend). This isolates conversation history per
    # user/session instead of every request sharing one global history, which
    # was the cause of context getting mixed together across chats.
    session_id = request.form.get("session_id") or "default"
    config = {"configurable": {"session_id": session_id}}

    persona_instruction = get_persona_instruction(persona)

    print("Message:", message)
    print("Area ID:", area_id)
    print("Parcel context:", parcel_context)
    print("Persona:", persona)
    print("Session:", session_id)

    response_text = ""

    if file:
        print("Uploaded file:", file)

        mime_type, image_base64 = image_to_base64(file)

        # Guardrail: reject images whose visual content is not agriculture
        # related before running the (expensive) analysis agent.
        if not is_image_agriculture_related(mime_type, image_base64):
            print("Image classified as NOT_AGRICULTURE_RELATED")
            return jsonify({"response": NOT_AGRICULTURE_RESPONSE})

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

        # Keep image turns in the per-session history so the user can ask
        # follow-up questions about the same image. History is isolated per
        # session (a fresh session_id per page load), so this does NOT mix
        # different users' or different chats' context. We skip the redundant
        # text-based classification router because the image already passed the
        # vision guardrail above, and feed the session's existing history in.
        session_history = get_history(session_id)

        result = agriculture_chain.invoke(
            {
                "messages": [human_message],
                "history": session_history.messages,
                "persona_instruction": persona_instruction,
            }
        )
        response_text = result.content

        # Persist a LIGHTWEIGHT version of this turn: the image plus the user's
        # question only, without the long analysis instructions. This way
        # follow-ups keep the image context while the history stays clean and
        # the big prompt is not repeated on every later turn.
        history_human = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": message if message else "Please analyze this uploaded field image.",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{image_base64}",
                },
            ]
        )
        session_history.add_message(history_human)
        session_history.add_message(result)

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
        result = with_history.invoke(
            {
                "messages": [human_message],
                "persona_instruction": persona_instruction
            },
            config=config,
        )
        response_text = result.content
    elif message:
        result = with_history.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "persona_instruction": persona_instruction
            },
            config=config,
        )
        response_text = result.content


    return jsonify({
        "response": response_text
    })

if __name__ == "__main__":
    app.run(debug=True)
