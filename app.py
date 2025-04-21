import streamlit as st
import json
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
# from azure.cosmos import CosmosClient, PartitionKey, exceptions # Commented out
from openai import OpenAI, RateLimitError, APIError # Import OpenAI client and specific errors

# --- Configuration & Initialization ---
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o") # Or "gpt-3.5-turbo", "gpt-4-turbo" etc.

# Cosmos DB Configuration (Commented Out)
# COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
# COSMOS_KEY = os.getenv("COSMOS_KEY")
# DATABASE_NAME = "AIChatLogs"
# CONTAINER_NAME = "Conversations"

# --- Load Prompts from Files ---

def load_prompt_from_file(filepath, default_prompt=""):
    """Loads prompt text from a file, with fallback."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Prompt file not found: {filepath}. Using default/empty prompt.")
        print(f"Error: Prompt file not found at {filepath}")
        return default_prompt
    except Exception as e:
        st.error(f"Error reading prompt file {filepath}: {e}")
        print(f"Error reading prompt file {filepath}: {e}")
        return default_prompt

# Load prompts into variables
# Ensure prompt_call_1.txt and prompt_call_2.txt exist and are correct
PROMPT_TEMPLATE_CALL_1_BASE = load_prompt_from_file("prompt_call_1.txt")
# Make sure prompt_call_2.txt contains the updated output instructions
PROMPT_TEMPLATE_CALL_2_BASE = load_prompt_from_file("prompt_call_2.txt")

# Check if prompts loaded successfully
if not PROMPT_TEMPLATE_CALL_1_BASE:
    st.stop("Failed to load critical prompt for Call 1. Please ensure 'prompt_call_1.txt' exists.")
if not PROMPT_TEMPLATE_CALL_2_BASE:
    st.stop("Failed to load critical prompt for Call 2. Please ensure 'prompt_call_2.txt' exists and includes the updated output instructions.")


# --- Helper Functions ---

# @st.cache_resource # Cache the client for efficiency - Re-enable if needed
def initialize_openai_client():
    """Initializes and returns the OpenAI client."""
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found in environment variables.")
        print("Error: OPENAI_API_KEY missing.")
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        print(f"OpenAI Client Error: {e}")
        return None

# Cosmos DB Functions (Commented Out)
# @st.cache_resource
# def initialize_cosmos_client(): ...
# def get_cosmos_container(_client): ...
# def log_to_cosmos(container, session_id, role, content): ...

def call_openai(client, messages, model=OPENAI_MODEL_NAME, temperature=0.2):
    """Calls the OpenAI API chat completion endpoint."""
    if not client:
        st.error("OpenAI client not initialized.")
        return None

    # Cosmos DB Logging Call (Commented Out)
    # log_to_cosmos(st.session_state.get('cosmos_container'), st.session_state.session_id, "user", json.dumps(messages[-1])) # Log last user message

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
        )
        response_content = chat_completion.choices[0].message.content

        # Cosmos DB Logging Call (Commented Out)
        # log_to_cosmos(st.session_state.get('cosmos_container'), st.session_state.session_id, "assistant", response_content)

        return response_content.strip()

    except RateLimitError as e:
        st.error(f"OpenAI API Rate Limit Error: {e}. Please check your plan and usage limits.")
        print(f"OpenAI Rate Limit Error: {e}")
        # Cosmos DB Logging Call (Commented Out)
        # log_to_cosmos(st.session_state.get('cosmos_container'), st.session_state.session_id, "system_error", f"OpenAI Rate Limit Error: {e}")
        return None
    except APIError as e:
        st.error(f"OpenAI API Error: {e}")
        print(f"OpenAI API Error: {e}")
        # Cosmos DB Logging Call (Commented Out)
        # log_to_cosmos(st.session_state.get('cosmos_container'), st.session_state.session_id, "system_error", f"OpenAI API Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred calling OpenAI: {e}")
        print(f"OpenAI Unexpected Error: {e}")
        # Cosmos DB Logging Call (Commented Out)
        # log_to_cosmos(st.session_state.get('cosmos_container'), st.session_state.session_id, "system_error", f"Unexpected OpenAI error: {e}")
        return None


def parse_generator_output(output_text):
    """Parses the output of LLM Call 1 based on headings."""
    # Basic parsing - might need refinement based on actual LLM output variations
    items_text = ""
    artifacts_text = ""
    try:
        # Ensure output_text is a string
        if not isinstance(output_text, str):
            print(f"Warning: Generator output is not a string: {type(output_text)}. Cannot parse.")
            return "", "" # Return empty strings if not parsable

        parts = output_text.split('## ')
        items_section = ""
        strategy_section = ""
        scenarios_section = ""
        cases_section = ""

        for part in parts:
            # Check startswith carefully
            part_lower = part.lower().strip() # Normalize for comparison
            if part_lower.startswith("extracted testable items"):
                items_section = "## " + part
            elif part_lower.startswith("draft test strategy"):
                 strategy_section = "## " + part
            elif part_lower.startswith("draft test scenarios"):
                 scenarios_section = "## " + part
            elif part_lower.startswith("draft test cases"):
                 cases_section = "## " + part

        items_text = items_section.strip()
        # Combine the draft artifacts for the next call
        artifacts_text = f"{strategy_section}\n\n{scenarios_section}\n\n{cases_section}".strip()

        if not items_text or not artifacts_text:
             print("Warning: Parsing Call 1 output might be incomplete based on expected headings.")
             # Fallback: return raw if parsing fails badly - adjust if needed
             # return output_text, output_text

        return items_text, artifacts_text
    except Exception as e:
        print(f"Error parsing generator output: {e}. Returning potentially raw/incomplete output.")
        # Fallback if splitting/parsing fails
        return output_text, output_text # Return raw output for both if parsing fails

def format_docs_for_prompt(uploaded_files):
    """Formats uploaded documents into a single string for the prompt."""
    content = []
    if not uploaded_files:
        return ""
    for doc in uploaded_files:
        # Ensure content is string
        doc_content = doc.get('content', '')
        if not isinstance(doc_content, str):
            doc_content = str(doc_content) # Attempt conversion if not string

        content.append(f"# Document Tag: {doc.get('tag', 'Unknown')}\n\n{doc_content}\n\n---\n")
    return "\n".join(content)

# format_chat_history_for_prompt is no longer needed as OpenAI takes the messages list directly

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="QA Test Artifact Generator (OpenAI)")
st.title("📄 QA Test Artifact Generator (using OpenAI)")

# Initialize OpenAI Client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = initialize_openai_client()

# Initialize Cosmos DB Client and Container (Commented Out)
# if 'cosmos_client' not in st.session_state:
#     st.session_state.cosmos_client = initialize_cosmos_client()
# if 'cosmos_container' not in st.session_state and st.session_state.get('cosmos_client'):
#     st.session_state.cosmos_container = get_cosmos_container(st.session_state.cosmos_client)
# elif 'cosmos_container' not in st.session_state:
#      st.session_state.cosmos_container = None

# Initialize session state variables
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    print(f"New session started: {st.session_state.session_id}")
if 'messages' not in st.session_state:
    # Initialize with system prompt if desired, or leave empty
    st.session_state.messages = [] # Stores chat history: [{"role": "user/assistant", "content": "..."}]
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = [] # Stores {'tag': '...', 'content': '...'}
if 'prompt1_text' not in st.session_state:
    # Initialize from file content loaded earlier
    st.session_state.prompt1_text = PROMPT_TEMPLATE_CALL_1_BASE
if 'generation_done' not in st.session_state:
    st.session_state.generation_done = False
if 'refined_output' not in st.session_state:
    st.session_state.refined_output = None # Stores the result of Call 2

# --- Sidebar for File Upload and Controls ---
with st.sidebar:
    st.header("📄 Document Upload")
    st.caption("Upload relevant documents one by one.")

    doc_tag = st.text_input("Document Tag/Name (e.g., BRS, Epic E-1)", key="doc_tag_input")
    uploaded_file = st.file_uploader("Choose a file (.txt, .md)", type=['txt', 'md'], key="file_uploader")

    if st.button("Add Document", key="add_doc_button"):
        if uploaded_file is not None and doc_tag:
            try:
                # Attempt decoding with fallback
                try:
                    content = uploaded_file.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    st.warning(f"UTF-8 decoding failed for {uploaded_file.name}. Trying latin-1.")
                    content = uploaded_file.getvalue().decode("latin-1")

                st.session_state.uploaded_files.append({"tag": doc_tag, "content": content})
                st.success(f"Added document: {doc_tag} ({uploaded_file.name})")
                # Clear inputs might require rerun or more complex state handling
            except Exception as e:
                st.error(f"Error reading file {uploaded_file.name}: {e}")
        elif not doc_tag:
            st.warning("Please enter a tag/name for the document.")
        else:
            st.warning("Please choose a file to upload.")

    st.subheader("Uploaded Documents:")
    if not st.session_state.uploaded_files:
        st.caption("No documents added yet.")
    else:
        for i, doc in enumerate(st.session_state.uploaded_files):
            st.markdown(f"- **{doc['tag']}**")

    st.divider()
    st.header("⚙️ Controls")
    if st.button("✨ New Chat Session", key="new_chat_button"):
        # Reset relevant session state variables
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.session_state.prompt1_text = PROMPT_TEMPLATE_CALL_1_BASE # Reload from file content
        st.session_state.generation_done = False
        st.session_state.refined_output = None
        st.session_state.session_id = str(uuid.uuid4()) # Start new session ID
        print(f"New session started: {st.session_state.session_id}")
        st.rerun() # Rerun the app to reflect the reset state

# --- Main Area ---

st.subheader("1. Initial Generation Prompt (Editable)")
st.caption("Review and edit the prompt for the initial artifact generation (LLM Call 1). Loaded from 'prompt_call_1.txt'.")
prompt1_edited = st.text_area(
    "Prompt for LLM Call 1:",
    value=st.session_state.prompt1_text,
    height=300,
    key="prompt1_editor"
)
# Update session state if edited
st.session_state.prompt1_text = prompt1_edited

if st.button("🚀 Generate Initial Artifacts", key="submit_button", disabled=(not st.session_state.openai_client)):
    if not st.session_state.openai_client:
         st.error("OpenAI client not initialized. Check API Key and environment variables.")
    elif not st.session_state.uploaded_files:
        st.warning("Please upload at least one document before generating.")
    else:
        st.session_state.generation_done = False # Reset flag
        st.session_state.messages = [] # Clear previous chat
        st.session_state.refined_output = None

        with st.spinner("Processing documents and calling OpenAI LLMs... (This may take a moment)"):
            # Prepare inputs
            tagged_docs_section = format_docs_for_prompt(st.session_state.uploaded_files)
            # Use the potentially edited prompt from the text area
            final_prompt_call_1_content = st.session_state.prompt1_text.format(tagged_docs_prompt_section=tagged_docs_section)

            # --- LLM Call 1 ---
            st.info("Initiating OpenAI LLM Call 1 (Generator)...")
            # OpenAI expects messages format
            messages_call_1 = [{"role": "user", "content": final_prompt_call_1_content}]
            generator_output = call_openai(st.session_state.openai_client, messages_call_1)

            if generator_output:
                st.info("OpenAI Call 1 completed. Parsing output...")
                # Parse output of Call 1
                items_text, artifacts_text = parse_generator_output(generator_output)

                if not items_text or not artifacts_text:
                     st.warning("Could not reliably parse the output from the first LLM call based on headings. Refinement quality may be affected.")
                     # Decide on fallback - using raw output might confuse Call 2
                     if not items_text: items_text = "Parsing Error: Could not find 'Extracted Testable Items'."
                     if not artifacts_text: artifacts_text = generator_output # Pass raw if artifact parsing failed

                # --- LLM Call 2 ---
                st.info("Initiating OpenAI LLM Call 2 (Refiner/Reviewer)...")
                # Ensure the prompt loaded from prompt_call_2.txt has the updated output instructions
                final_prompt_call_2_content = PROMPT_TEMPLATE_CALL_2_BASE.format(
                    testable_items_list_text=items_text,
                    draft_test_artifacts_text=artifacts_text,
                    tagged_docs_prompt_section=tagged_docs_section
                )
                messages_call_2 = [{"role": "user", "content": final_prompt_call_2_content}]
                refined_output_text = call_openai(st.session_state.openai_client, messages_call_2)

                if refined_output_text:
                    st.info("OpenAI Call 2 completed.")
                    st.session_state.refined_output = refined_output_text
                    # Add the refined output as the first "assistant" message in the chat history
                    # This output *now includes* the 'Final Testable Items List' heading and content
                    st.session_state.messages = [{"role": "assistant", "content": refined_output_text}]
                    st.session_state.generation_done = True
                    st.success("Initial artifact generation and refinement complete!")
                    st.rerun() # Rerun to display the chat history
                else:
                    st.error("OpenAI LLM Call 2 (Refiner) failed.")
            else:
                st.error("OpenAI LLM Call 1 (Generator) failed.")


st.subheader("2. Generated Artifacts & Chat")

# Display chat messages from history
if st.session_state.generation_done or st.session_state.messages:
     if st.session_state.generation_done and not st.session_state.messages:
         # If generation just finished but rerun hasn't populated messages list yet
         st.info("Processing complete. Displaying results...")
     elif st.session_state.messages:
        # The first message now contains the items list, strategy, scenarios, cases, matrix
        st.info("Review the generated artifacts below (including Testable Items). Use the chat box for follow-up questions or modifications.")
        chat_container = st.container(height=500, border=True) # Make chat scrollable
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    # Use markdown for better formatting of headings etc.
                    st.markdown(message["content"], unsafe_allow_html=True)

# Chat input for follow-up questions
if prompt := st.chat_input("Ask follow-up questions or request modifications...", disabled=(not st.session_state.openai_client)):
    if not st.session_state.openai_client:
         st.error("OpenAI client not initialized. Cannot process chat.")
    elif not st.session_state.generation_done:
        st.warning("Please generate the initial artifacts first using the 'Generate Initial Artifacts' button.")
    else:
        # Add user message to state and display immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare message history for OpenAI API call
        # The entire conversation history (including the initial detailed assistant message) is needed
        messages_for_api = st.session_state.messages

        # Call LLM for follow-up
        with st.spinner("OpenAI is thinking..."):
            response = call_openai(
                st.session_state.openai_client,
                messages=messages_for_api
            )

            # Display assistant response
            with st.chat_message("assistant"):
                if response:
                    st.markdown(response, unsafe_allow_html=True)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Failed to get response from the assistant.")
                    # Add error message to history to indicate failure
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error processing your request."})

        # Rerun may be needed if state updates within the 'with' block aren't immediately reflected
        # outside it before the next interaction, especially for the message list persistence.
        # st.rerun() # Test if needed for chat history updates