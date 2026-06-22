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
import os
import urllib.request
import urllib.parse
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
    model="gemini-2.5-pro",
    # Auto-retry transient failures (network/DNS blips, rate limits) so a
    # single hiccup reaching the Gemini API doesn't fail the whole request.
    max_retries=3,
    timeout=60,
)

# region storing
store = {}
# endregion


# region Airtable feedback integration
# Credentials come from the environment (never hard-code the API key). Set
# AIRTABLE_API_KEY (and optionally AIRTABLE_BASE_ID / AIRTABLE_TABLE_NAME) in
# your .env locally and in the host's variables when deploying.
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appdQPElA6mOgh8EJ")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "CRM")

# Column names in the Airtable feedback table. Kept in one place so a rename in
# Airtable only needs a one-line change here (or an env override).
AIRTABLE_RATING_FIELD = os.getenv("AIRTABLE_RATING_FIELD", "rey bolmesi")
AIRTABLE_REVIEW_FIELD = os.getenv("AIRTABLE_REVIEW_FIELD", "Review")
AIRTABLE_STATUS_FIELD = os.getenv("AIRTABLE_STATUS_FIELD", "Status")
# Every new piece of feedback starts as "not reviewed yet".
AIRTABLE_DEFAULT_STATUS = os.getenv("AIRTABLE_DEFAULT_STATUS", "baxilmadi")


def save_feedback_to_airtable(rating: str, review: str,
                              status: str = AIRTABLE_DEFAULT_STATUS) -> dict:
    """Create one feedback record in Airtable (rating + Review + Status)."""
    if not AIRTABLE_API_KEY:
        raise RuntimeError("AIRTABLE_API_KEY is not set in the environment.")

    url = (
        "https://api.airtable.com/v0/"
        f"{AIRTABLE_BASE_ID}/{urllib.parse.quote(AIRTABLE_TABLE_NAME)}"
    )
    payload = json.dumps(
        {
            # typecast lets Airtable accept the Status value even if the option
            # casing differs slightly, instead of rejecting the whole request.
            "typecast": True,
            "records": [
                {
                    "fields": {
                        AIRTABLE_RATING_FIELD: rating,
                        AIRTABLE_REVIEW_FIELD: review,
                        AIRTABLE_STATUS_FIELD: status,
                    }
                }
            ],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))
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

    "agronomist": """
    You are an experienced local farmer and you have 25 years experience.Do not use unnecessary words..Use a formal and professional tone..If question is in Azerbaijan, then answer in Azerbaijan, otherwise answer in English.

""",

    "farmer": """
You are a local farmer.Use a formal and professional tone.Do not use unnecessary words..If question is in Azerbaijan, then answer in Azerbaijan, otherwise answer in English.
"""
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
# ROLE & IDENTITY
You are AgroVision AI, an elite, result-oriented agricultural intelligence assistant with deep expertise in agronomy, plant pathology, and remote sensing. Your primary mission is to provide actionable, high-precision insights that farmers and agronomists can use immediately to save crops and optimize yields.

# USER CONTEXT & METADATA (CRUCIAL & CONDITIONAL LOGIC)
You may or may not be provided with a JSON object representing the user's profile and registered agricultural fields. You must dynamically adapt your logic based on the presence of this metadata:

- **CASE A: IF JSON METADATA IS PROVIDED:**
  1. Actively cross-reference the visual content of the image with the JSON data (`current_crop`, `average_ndvi`, `average_ndwi`).
  2. Evaluate field and crop compatibility.
  3. Include "## 🗺️ 1. Ərazi və Bitki Uyğunluğu (Field & Crop Verification)" as the very first section in your output.

- **CASE B: IF JSON METADATA IS NOT PROVIDED (OR IS EMPTY/NULL):**
  1. Skip the verification step entirely. Do not invent or assume any field data.
  2. Treat the image as a general agricultural sample.
  3. DO NOT include "## 🗺️ 1. Ərazi və Bitki Uyğunluğu" section in your output. Start your response directly from "## 📊 2. Şəkil Tipi və Təsnifat".

# LANGUAGE PROTOCOL
1. Always analyze and respond in the exact language used by the user (e.g., Azerbaijani, English, Russian).
2. CRUCIAL RULE: If the user uploads an image WITHOUT any text, your default language MUST be Azerbaijani.

# CORE OPERATIONAL MANDATE (RESULT-ORIENTED)
- Do not be overly defensive. Instead of saying "I cannot know from one image," provide the "Most Probable Diagnosis" based on visual evidence, ranking the likelihood.
- Be decisive. Farmers need quick, practical solutions, not academic disclaimers.

# ANALYSIS & CLASSIFICATION STEPS
1. **CHECK CONTEXT:** Detect if User JSON Metadata is present.
2. **VERIFY (Only for CASE A):** Compare image content with metadata metrics.
3. **CLASSIFY (All Cases):** Determine image scope: [Ərazi/Dron/Peyk] OR [Bağ/Plantasiya] OR [Spesifik Bitki/Yarpaq/Meyvə/Torpaq].
4. **DIAGNOSE (All Cases):** Identify visual anomalies. If metadata is present, correlate anomalies with NDVI/NDWI values.
5. **PRESCRIBE (All Cases):** Formulate immediate agronomic solutions.

# PREFERRED OUTPUT STRUCTURE (STRICTLY ENFORCED)
Depending on the conditional logic above, format your response using the following template:

[INCLUDE THIS SECTION ONLY FOR CASE A - IF JSON METADATA IS PRESENT]
## 🗺️  Ərazi və Bitki Uyğunluğu (Field & Crop Verification)
* **İstifadəçi Sahəsi:** [Sahənin adı (məs: North Wheat Field) və sistemdə qeyd olunan cari bitki]
* **Vizual Uyğunluq Statusu:** [BƏLİ / XEYR / ŞÜBHƏLİ] — (Şəkildəki bitkinin profilinizdəki bitki növünə uyğun olub-olmadığını yazın. Məsələn: "Bəli, şəkildəki bitki sahənizdə qeyd olunan Qarğıdalı (Corn) bitkisidir" və ya "Xeyr, sahənizdə Buğda qeyd olunub, lakin şəkildə Pomidor yarpağı görünür").
* **Göstərici Sintezi (NDVI/NDWI):** [Metadatadakı spektral dəyərlərin şəkildəki vizual vəziyyətlə (məs: NDVI 0.68-dir, bu da orta sıxlığı göstərir, lakin şəkildəki lokal saralma xəstəliyin başlanğıcı ola bilər) qısa əlaqəsi].

[START HERE FOR ALL CASES - IF NO METADATA, THIS IS THE FIRST SECTION]
## 📊  Şəkil Tipi və Təsnifat (Image Type & Classification)
* **Təsnifat:** [Ərazi / Bağ-Plantasiya / Spesifik Bitki (Yarpaq, Meyvə, Gövdə) / Torpaq]
* **Müşahidə Olunan Faktlar:** [Şəkildə görünən vizual detalları (məs: saralma, ləkələr, seyrəklik, alaqlı otlar) 1-2 cümlə ilə qeyd edin].

## 🔍 Aqronomik Diaqnoz (Agronomic Diagnosis)
* **Ən Ehtimal Olunan Problem:** [Məsələn: Azot çatışmazlığı / Alternaria xəstəliyi / Su stresi / Dron xəritəsində qeyri-bərabər inkişaf]
* **Əminlik Səviyyəsi:** [Yüksək / Orta / Aşağı] — (Səbəbini vizual markerlə qısa əsaslandırın).
* **Alternativ Səbəblər:** [Əgər varsa, digər 1 mümkün səbəb].

## ⚡ Təcili Fəaliyyət Planı (Actionable Next Steps)
* **Sahədə İlkin Yoxlama:** [Aqronomun sahəyə gedəndə ilk baxmalı olduğu nöqtə və ya etməli olduğu fiziki test].
* **Təklif Olunan Həll (Müalicə/İdarəetmə):** [Problemin həlli üçün konkret aqronomik addım: suvarma tənzimlənməsi, spesifik gübrə növü, funqisid/insektisid tətbiqi və ya idarəetmə zonalarının ayrılması].

## 📌 Dəqiq Diaqnoz Üçün Lazım Olan Data (Missing Critical Data)
* [Diaqnozu 100% dəqiqləşdirmək üçün lazım olan tək bir kritik məlumat, məsələn: torpaq rütubəti, son gübrələmə tarixi və ya dəqiq temperatur].
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


def build_area_list():
    """Flatten database.json into a flat list of areas for the frontend dropdown.

    This is the single source of truth: the sidebar selector and the chat
    context both come from database.json, so they can never drift apart the way
    a hard-coded frontend list does.
    """
    areas = []
    for user in db["users"]:
        for area in user["saved_areas"]:
            metrics = area.get("baseline_metrics", {})
            areas.append(
                {
                    "area_id": area["area_id"],
                    "area_name": area.get("area_name", area["area_id"]),
                    "current_crop": area.get("current_crop", ""),
                    "average_ndvi": metrics.get("average_ndvi"),
                    "average_ndwi": metrics.get("average_ndwi"),
                    "user_name": user.get("name", ""),
                }
            )
    return areas


def find_area_by_id(area_id):
    """Return the full saved_area record (with coordinates) for an area_id."""
    return next(
        (
            area
            for user in db["users"]
            for area in user["saved_areas"]
            if area["area_id"] == area_id
        ),
        None,
    )


def build_spatial_context(area_id):
    """Build the system-prompt spatial-context block for the selected area.

    Returns "" when no/unknown area is given so the model is told nothing rather
    than being fed a stale or wrong field. The result is injected into the
    system prompt for the current turn only (never stored in chat history).
    """
    if not area_id:
        return ""

    area_info = find_area_by_id(area_id)
    if area_info is None:
        return ""

    return (
        "ACTIVE FIELD CONTEXT (from the spatial database). "
        "This is the ONLY field the user is currently asking about — "
        "ignore any field mentioned earlier in the conversation:\n"
        f"{json.dumps(area_info, ensure_ascii=False)}\n\n"
        "Use this data to answer questions about this user's field, parcel, "
        "area_id, current crop, coordinates, NDVI, and NDWI. If the user asks "
        "about something not present in this data, say so instead of guessing."
    )

# region prompt classification and agriculture agent
prompt_classification_agent = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_classification_agent),
        MessagesPlaceholder(variable_name="messages"),

    ]
)

prompt_agriculture_agent = ChatPromptTemplate.from_messages(
    [
        # spatial_context is injected into the SYSTEM message for the current
        # turn only. It is intentionally NOT part of the stored chat history, so
        # the active area never gets duplicated across turns and a previously
        # selected area can't leak into answers about a new one.
        (
            "system",
            system_prompt_agriculture_agent
            + "\n\n{persona_instruction}\n\n{spatial_context}",
        ),
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


@app.route("/areas", methods=["GET"])
def areas():
    """Serve the list of saved areas from database.json to the frontend.

    The sidebar dropdown loads from here on page load, so adding/editing a
    field in database.json is enough — no frontend change is needed.
    """
    return jsonify({"areas": build_area_list()})

@app.route("/chat", methods=["POST"])
def chat():
    # Wrap the handler so transient failures (e.g. network/DNS blips reaching
    # the Gemini API: httpx.ConnectError / getaddrinfo failed) return a clean
    # message to the UI instead of a raw 500 traceback.
    try:
        return _chat_impl()
    except Exception as e:
        print("Chat error:", repr(e))
        return jsonify({
            "response": "Bağlantı xətası oldu (şəbəkə/DNS problemi ola bilər). "
                        "Zəhmət olmasa bir azdan yenidən cəhd et."
        }), 200


def _chat_impl():
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

        # If a tracking area is selected, give the image analyzer that field's
        # data too (Field & Crop Verification). Otherwise leave it empty.
        spatial_context = build_spatial_context(area_id) if area_id else ""

        result = agriculture_chain.invoke(
            {
                "messages": [human_message],
                "history": session_history.messages,
                "persona_instruction": persona_instruction,
                "spatial_context": spatial_context,
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
        # The area's data is injected into the system prompt for THIS turn only
        # (via spatial_context), not stored in history. Only the user's plain
        # question goes into history, so switching areas later never mixes an
        # old field's data into a new answer and the context isn't duplicated
        # on every turn.
        spatial_context = build_spatial_context(area_id)
        result = with_history.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "persona_instruction": persona_instruction,
                "spatial_context": spatial_context,
            },
            config=config,
        )
        response_text = result.content
    elif message:
        result = with_history.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "persona_instruction": persona_instruction,
                "spatial_context": "",
            },
            config=config,
        )
        response_text = result.content


    return jsonify({
        "response": response_text
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    # The frontend asks the user "how was this answer?" after a few replies.
    # rating  -> stored in the "name" column (e.g. "👍 Good" / "👎 Bad")
    # comment -> stored in the "Review" column (free-text feedback)
    rating = (request.form.get("rating") or "").strip()
    comment = (request.form.get("comment") or "").strip()
    session_id = request.form.get("session_id") or "default"

    print("Feedback:", {"rating": rating, "comment": comment, "session": session_id})

    if not rating and not comment:
        return jsonify({"ok": False, "error": "Empty feedback"}), 400

    name = rating or "Feedback"
    review = comment or "(no comment)"

    try:
        result = save_feedback_to_airtable(name, review)
        record_id = (result.get("records") or [{}])[0].get("id")
        return jsonify({"ok": True, "record_id": record_id})
    except Exception as e:
        print("Airtable feedback error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
