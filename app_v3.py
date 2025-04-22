import streamlit as st
import json
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests
import logging # Import the logging library

# --- Logging Configuration ---
# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Output to console
        # You can add logging.FileHandler("app.log") here to also log to a file
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration & Initialization ---
load_dotenv()
logger.info("Environment variables loaded.")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:1b") # Default model
logger.info(f"Ollama configured: URL={OLLAMA_BASE_URL}, Model={OLLAMA_MODEL_NAME}")

# --- Load Prompts from Files ---
def load_prompt_from_file(filepath, default_prompt=""):
    logger.info(f"Attempting to load prompt from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Successfully loaded prompt from: {filepath}")
            return content
    except FileNotFoundError:
        st.error(f"Prompt file not found: {filepath}. Using default/empty prompt.")
        logger.error(f"Prompt file not found at {filepath}")
        return default_prompt
    except Exception as e:
        st.error(f"Error reading prompt file {filepath}: {e}")
        logger.error(f"Error reading prompt file {filepath}: {e}")
        return default_prompt

PROMPT_TEMPLATE_CALL_1_BASE = load_prompt_from_file("prompt_call_1.txt")
PROMPT_TEMPLATE_CALL_2_BASE = load_prompt_from_file("prompt_call_2.txt")

if not PROMPT_TEMPLATE_CALL_1_BASE:
    logger.critical("Failed to load critical prompt for Call 1. Exiting.")
    st.stop("Failed to load critical prompt for Call 1. Please ensure 'prompt_call_1.txt' exists.")
if not PROMPT_TEMPLATE_CALL_2_BASE:
    logger.critical("Failed to load critical prompt for Call 2. Exiting.")
    st.stop("Failed to load critical prompt for Call 2. Please ensure 'prompt_call_2.txt' exists and includes the updated output instructions.")

# --- Helper Functions ---
def call_llm_service(prompt: str) -> str | None:
    logger.info(f"Sending prompt to Ollama Service ({OLLAMA_MODEL_NAME}). Prompt length: {len(prompt)}")
    # logger.debug(f"Full prompt being sent:\n{prompt}") # Uncomment for verbose prompt logging

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(
            OLLAMA_API_ENDPOINT,
            headers=headers,
            data=json.dumps(payload),
            timeout=3000 # Increased timeout
        )
        response.raise_for_status()
        response_data = response.json()
        llm_response = response_data.get("response", "")
        logger.info(f"Received response from Ollama. Response length: {len(llm_response)}")
        # logger.debug(f"Full response received:\n{llm_response}") # Uncomment for verbose response logging
        return llm_response.strip()
    except requests.exceptions.ConnectionError as e:
        st.error(f"Connection Error: Failed to connect to Ollama at {OLLAMA_API_ENDPOINT}. Is Ollama running?")
        logger.error(f"Ollama Connection Error: {e}")
        return None
    except requests.exceptions.Timeout as e:
        st.error(f"Timeout Error: The request to Ollama timed out (300s).")
        logger.error(f"Ollama Timeout Error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama service: {e}")
        logger.error(f"Ollama Request Error: {e}")
        return None
    except json.JSONDecodeError as e:
         st.error(f"Failed to decode JSON response from Ollama: {e}")
         logger.error(f"Ollama JSON Decode Error: {e}. Response text: {response.text[:500] if 'response' in locals() and hasattr(response, 'text') else 'N/A'}...")
         return None
    except Exception as e:
        st.error(f"An unexpected error occurred calling local LLM: {e}")
        logger.error(f"Ollama Unexpected Error: {e}")
        return None

def parse_generator_output(output_text):
    logger.info("Parsing output from LLM Call 1 (Generator).")
    items_text = ""
    artifacts_text = ""
    try:
        if not isinstance(output_text, str):
            logger.warning(f"Generator output is not a string: {type(output_text)}. Cannot parse.")
            return "", ""
        parts = output_text.split('## ')
        items_section, strategy_section, scenarios_section, cases_section = "", "", "", ""
        for part in parts:
            part_lower = part.lower().strip()
            if part_lower.startswith("extracted testable items"): items_section = "## " + part
            elif part_lower.startswith("draft test strategy"): strategy_section = "## " + part
            elif part_lower.startswith("draft test scenarios"): scenarios_section = "## " + part
            elif part_lower.startswith("draft test cases"): cases_section = "## " + part
        items_text = items_section.strip()
        artifacts_text = f"{strategy_section}\n\n{scenarios_section}\n\n{cases_section}".strip()
        if not items_text or not artifacts_text:
             logger.warning("Parsing Call 1 output might be incomplete based on expected headings.")
        else:
            logger.info("Successfully parsed generator output.")
        return items_text, artifacts_text
    except Exception as e:
        logger.error(f"Error parsing generator output: {e}. Returning potentially raw/incomplete output.")
        return output_text, output_text

def format_docs_for_prompt(uploaded_files):
    logger.info(f"Formatting {len(uploaded_files)} uploaded documents for prompt.")
    content = []
    if not uploaded_files: return ""
    for doc in uploaded_files:
        doc_content = doc.get('content', '')
        if not isinstance(doc_content, str): doc_content = str(doc_content)
        content.append(f"# Document Tag: {doc.get('tag', 'Unknown')}\n\n{doc_content}\n\n---\n")
    formatted_docs = "\n".join(content)
    logger.info(f"Formatted documents string length: {len(formatted_docs)}")
    return formatted_docs

def format_history_for_local_llm(messages):
    logger.info(f"Formatting chat history of {len(messages)} messages for local LLM.")
    prompt_string = ""
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content_msg = msg.get("content", "")
        prompt_string += f"{role}:\n{content_msg}\n\n"
    prompt_string += "Assistant:\n"
    logger.info(f"Formatted history string length: {len(prompt_string)}")
    return prompt_string

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="QA Test Artifact Generator")

# Center align title using Markdown and HTML
st.markdown("<h1 style='text-align: center;'>QA Test Artifact Generator</h1>", unsafe_allow_html=True)
# st.title("QA Test Artifact Generator") # Original title, replaced by markdown for centering

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"New session started: {st.session_state.session_id}")
if 'messages' not in st.session_state: st.session_state.messages = []
if 'uploaded_files' not in st.session_state: st.session_state.uploaded_files = []
if 'prompt1_text' not in st.session_state: st.session_state.prompt1_text = PROMPT_TEMPLATE_CALL_1_BASE
if 'generation_done' not in st.session_state: st.session_state.generation_done = False
if 'refined_output' not in st.session_state: st.session_state.refined_output = None

with st.sidebar:
    st.header("📄 Document Upload")
    st.caption("Upload relevant documents one by one.")
    doc_tag = st.text_input("Document Tag/Name (e.g., BRS, Epic E-1)", key="doc_tag_input")
    uploaded_file = st.file_uploader("Choose a file (.txt, .md)", type=['txt', 'md'], key="file_uploader")
    if st.button("Add Document", key="add_doc_button"):
        if uploaded_file is not None and doc_tag:
            try:
                try: content = uploaded_file.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning(f"UTF-8 decoding failed for {uploaded_file.name}. Trying latin-1.")
                    content = uploaded_file.getvalue().decode("latin-1")
                st.session_state.uploaded_files.append({"tag": doc_tag, "content": content})
                logger.info(f"Added document: {doc_tag} ({uploaded_file.name})")
                st.success(f"Added document: {doc_tag} ({uploaded_file.name})")
            except Exception as e:
                logger.error(f"Error reading file {uploaded_file.name}: {e}")
                st.error(f"Error reading file {uploaded_file.name}: {e}")
        elif not doc_tag: st.warning("Please enter a tag/name for the document.")
        else: st.warning("Please choose a file to upload.")
    st.subheader("Uploaded Documents:")
    if not st.session_state.uploaded_files: st.caption("No documents added yet.")
    else:
        for i, doc in enumerate(st.session_state.uploaded_files): st.markdown(f"- **{doc['tag']}**")
    st.divider()
    # st.header("⚙️ Controls") # Removed "Controls" label
    if st.button("✨ New Chat Session", key="new_chat_button"):
        logger.info("New chat session initiated by user.")
        st.session_state.messages, st.session_state.uploaded_files = [], []
        st.session_state.prompt1_text = PROMPT_TEMPLATE_CALL_1_BASE
        st.session_state.generation_done, st.session_state.refined_output = False, None
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"New session ID: {st.session_state.session_id}")
        st.rerun()

st.subheader("Pre-defined Generation Prompt (Editable)")
st.caption("Review and edit the prompt for the initial artifact generation (LLM Call 1). Loaded from 'prompt_call_1.txt'.")
prompt1_edited = st.text_area("Prompt for LLM Call 1:", value=st.session_state.prompt1_text, height=300, key="prompt1_editor")
st.session_state.prompt1_text = prompt1_edited

if st.button("Generate Initial Artifacts", key="submit_button"):
    logger.info("'Generate Initial Artifacts' button clicked.")
    if not st.session_state.uploaded_files:
        st.warning("Please upload at least one document before generating.")
        logger.warning("Generation attempt without uploaded documents.")
    else:
        st.session_state.generation_done, st.session_state.messages, st.session_state.refined_output = False, [], None
        logger.info("Starting artifact generation process.")
        with st.spinner("Processing documents and calling Local LLM... (Time varies based on model)"):
            tagged_docs_section = format_docs_for_prompt(st.session_state.uploaded_files)
            final_prompt_call_1_content = st.session_state.prompt1_text.format(tagged_docs_prompt_section=tagged_docs_section)
            logger.info("Initiating Local LLM Call 1 (Generator).")
            generator_output = call_llm_service(final_prompt_call_1_content)
            if generator_output:
                logger.info("Local LLM Call 1 completed.")
                items_text, artifacts_text = parse_generator_output(generator_output)
                if not items_text or not artifacts_text:
                     logger.warning("Parsing Call 1 output was incomplete. Refinement quality may be affected.")
                     if not items_text: items_text = "Parsing Error: Could not find 'Extracted Testable Items'."
                     if not artifacts_text: artifacts_text = generator_output
                logger.info("Initiating Local LLM Call 2 (Refiner/Reviewer).")
                final_prompt_call_2_content = PROMPT_TEMPLATE_CALL_2_BASE.format(
                    testable_items_list_text=items_text,
                    draft_test_artifacts_text=artifacts_text,
                    tagged_docs_prompt_section=tagged_docs_section
                )
                refined_output_text = call_llm_service(final_prompt_call_2_content)
                if refined_output_text:
                    logger.info("Local LLM Call 2 completed.")
                    st.session_state.refined_output = refined_output_text
                    st.session_state.messages = [{"role": "assistant", "content": refined_output_text}]
                    st.session_state.generation_done = True
                    st.success("Initial artifact generation and refinement complete!")
                    logger.info("Artifact generation and refinement successful.")
                    st.rerun()
                else:
                    st.error("Local LLM Call 2 (Refiner) failed.")
                    logger.error("Local LLM Call 2 (Refiner) failed.")
            else:
                st.error("Local LLM Call 1 (Generator) failed.")
                logger.error("Local LLM Call 1 (Generator) failed.")

st.subheader("Generated Artifacts") # section title
# Check if generation has been completed OR if there are messages (meaning generation was done before)
if st.session_state.generation_done or st.session_state.messages:
    # --- Create the container ONLY when artifacts are ready or being processed ---
    artifact_container = st.container(height=500, border=True)
    with artifact_container:
        # If messages exist and the first one is the assistant's output (the generated artifacts)
        if st.session_state.messages and st.session_state.messages[0]["role"] == "assistant":
            # Display the generated artifacts inside the scrollable container
            st.markdown(st.session_state.messages[0]["content"], unsafe_allow_html=True)
        elif st.session_state.generation_done and not st.session_state.messages:
            # Handles the brief moment after generation finishes but before rerun populates messages
            # Display temporary message inside the container
            st.info("Processing complete. Artifacts are being prepared...")
        else:
            # Fallback caption inside the container if something unexpected happens after generation_done=True
            # This case should ideally not be reached if logic is sound.
            st.caption("Generated artifacts will appear here once generation is complete.")

else:
    # --- Display placeholder caption directly under the subheader BEFORE generation ---
    st.caption("Generated artifacts (Testable Items, Strategy, Scenarios, Cases, Matrix) will be displayed here once you click 'Generate Initial Artifacts'.")
st.subheader("ChatBox") # New section for Chat
if st.session_state.generation_done: # Only show chat if generation is done
    # Display subsequent chat messages (user questions and assistant replies)
    if len(st.session_state.messages) > 1: # If there are follow-up messages
        chat_history_container = st.container(height=300, border=True)
        with chat_history_container:
            for i, message in enumerate(st.session_state.messages):
                if i == 0: continue # Skip the first refined output, already shown above
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)

    # Chat input with a label
    user_chat_prompt = st.chat_input("Ask follow-up questions or request modifications...")
    if user_chat_prompt:
        logger.info(f"User follow-up question: {user_chat_prompt}")
        st.session_state.messages.append({"role": "user", "content": user_chat_prompt})
        # Display user message immediately in the chat history section if it's separate
        # For now, it will appear on next rerun after assistant responds

        history_prompt_string = format_history_for_local_llm(st.session_state.messages)
        with st.spinner("Local LLM is thinking..."):
            logger.info("Sending follow-up to Local LLM.")
            response = call_llm_service(prompt=history_prompt_string)
            if response:
                logger.info("Received follow-up response from Local LLM.")
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                logger.error("Failed to get follow-up response from Local LLM.")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error processing your request."})
        st.rerun() # Rerun to display the new messages
else:
    st.caption("Generate artifacts first to enable the ChatBox.")
