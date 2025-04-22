# QA Test Artifact Generator (Streamlit + Local LLM)

## Description

This is a Streamlit web application designed to assist Quality Assurance (QA) teams in generating test artifacts based on requirement documents. Users can upload documents like Business Requirement Specifications (BRS), Epics, User Stories, etc. The application then utilizes a locally running Large Language Model (LLM) via Ollama to generate:

1.  A list of **Testable Items** extracted from the documents.
2.  A **Test Strategy**.
3.  **Test Scenarios**.
4.  Detailed **Test Cases** (including positive, negative, and boundary conditions).
5.  A **Traceability Matrix** linking Testable Items to Test Cases.

The generation process involves a two-stage LLM approach (Generator + Refiner) for improved quality. After the initial generation, users can interact with the LLM through a chat interface to request modifications, additions, or clarifications.

## Prerequisites

1.  **Python:** Version 3.9 or higher recommended.
2.  **Ollama:** You need Ollama installed and running locally. Download from [https://ollama.com/](https://ollama.com/).
3.  **Ollama Model:** Pull the specific LLM you intend to use via the Ollama CLI. This code defaults to `llama3.2:1b`.
    ```bash
    ollama pull llama3.2:1b
    ```
    *(Replace `llama3.2:1b` if using a different model)*
4.  **Python Libraries:** Install the required libraries:
    ```bash
    pip install streamlit python-dotenv requests
    ```
5.  **Environment File (`.env`):** Create a file named `.env` in the root directory of the project with the following variables:
    ```dotenv
    OLLAMA_BASE_URL="http://localhost:11434" # Adjust if Ollama runs on a different port/host
    OLLAMA_MODEL_NAME="llama3.2:1b"        # Must match the model pulled via Ollama
    ```
6.  **Prompt Files:** Create two text files in the root directory:
    *   `prompt_call_1.txt`: Contains the detailed instructions for the first LLM call (Generator). See previous code examples for the required content.
    *   `prompt_call_2.txt`: Contains the detailed instructions for the second LLM call (Refiner), including the final output structure. See previous code examples for the required content (ensure it asks for the final, non-"Refined" headings and includes the "Testable Items List" in the output).

## How to Run

1.  Ensure Ollama is running in the background with the specified model available (`ollama list` to check).
2.  Make sure the `.env` file and the two prompt files (`prompt_call_1.txt`, `prompt_call_2.txt`) are correctly set up in the project's root directory.
3.  Navigate to the project's root directory in your terminal.
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
5.  The application should open in your web browser.

## Overall Workflow

The application follows these steps:

1.  **Initialization:** Loads environment variables, configurations (Ollama URL, model name), and prompt templates from text files. Initializes the Streamlit UI elements and session state variables. Sets up logging.
2.  **User Input:**
    *   The user uploads requirement documents (e.g., `.txt`, `.md`) via the sidebar, providing a descriptive tag for each.
    *   The user can review and optionally edit the pre-defined prompt for the first LLM call in the main text area.
3.  **Generation Trigger:** The user clicks the "Generate Initial Artifacts" button.
4.  **Document Formatting:** The content of all uploaded documents is combined into a single string, with each document clearly marked by its tag.
5.  **LLM Call 1 (Generator):**
    *   The formatted documents are inserted into the (potentially edited) Prompt 1 template.
    *   This combined prompt is sent as a single string to the configured local Ollama model via the `/api/generate` endpoint using the `call_llm_service` function.
    *   The goal of this call is to extract an initial list of "Testable Items" and draft the Strategy, Scenarios, and Test Cases.
6.  **Parsing:** The text response from Call 1 is parsed by the `parse_generator_output` function to separate the "Extracted Testable Items" list from the other draft artifacts.
7.  **LLM Call 2 (Refiner):**
    *   The Prompt 2 template is formatted with:
        *   The "Extracted Testable Items" list from Call 1.
        *   The draft artifacts (Strategy, Scenarios, Cases) from Call 1.
        *   The original formatted documents string (for cross-referencing and completeness).
    *   This combined prompt is sent as a single string to the Ollama model via `call_llm_service`.
    *   The goal of this call is to review the draft, use the original documents to ensure completeness (especially for Testable Items), refine the artifacts, and generate the final output including the Traceability Matrix. The output should follow the structure defined in `prompt_call_2.txt` (e.g., `## Testable Items List`, `## Test Strategy`, etc.).
8.  **Display Initial Results:**
    *   The complete text response from Call 2 is stored.
    *   This response (containing all final artifacts) is displayed in the "Generated Artifacts" section within a scrollable container.
9.  **Conversational Follow-up (Chat):**
    *   The "ChatBox" section becomes active.
    *   The user types a follow-up question or modification request.
    *   The *entire* conversation history (starting with the full output from Call 2, followed by all subsequent user/assistant turns) is formatted into a single prompt string by `format_history_for_local_llm`.
    *   This history prompt string is sent to the Ollama model via `call_llm_service`.
    *   The LLM generates a response based on the conversational context.
    *   The user's question and the LLM's response are added to the chat display and the history list. This loop continues for further refinement.
10. **New Session:** The user can click "New Chat Session" in the sidebar to clear all uploaded documents, chat history, and generated artifacts, allowing them to start fresh.

## Code Breakdown (`app.py`)

1.  **Imports and Logging Setup:** Imports necessary libraries (`streamlit`, `requests`, `dotenv`, `logging`, etc.) and configures basic logging to the console.
2.  **Configuration & Initialization:** Loads `.env` variables, sets Ollama connection details (`OLLAMA_BASE_URL`, `OLLAMA_API_ENDPOINT`, `OLLAMA_MODEL_NAME`).
3.  **Load Prompts from Files:**
    *   `load_prompt_from_file()`: Safely reads prompt text from specified files (`prompt_call_1.txt`, `prompt_call_2.txt`), handles errors, and logs activity. The application stops if essential prompts cannot be loaded.
4.  **Helper Functions:**
    *   `call_llm_service(prompt)`: Sends the provided `prompt` string to the configured Ollama `/api/generate` endpoint using `requests.post`. Handles JSON payload creation, API response parsing (extracting the "response" field), error handling (connection, timeout, HTTP errors, JSON errors), and logging. Returns the LLM response string or `None`.
    *   `parse_generator_output(output_text)`: Takes the raw text output from LLM Call 1. Splits the text based on `## ` headings to separate the extracted testable items from the draft strategy, scenarios, and cases. Returns two strings. Includes basic error handling.
    *   `format_docs_for_prompt(uploaded_files)`: Takes the list of uploaded file dictionaries. Creates a single formatted string containing the content of all documents, each preceded by its user-defined tag (e.g., `# Document Tag: BRS`).
    *   `format_history_for_local_llm(messages)`: Takes the list of chat history messages (dictionaries with "role" and "content"). Converts this list into a single string suitable for the Ollama `/api/generate` endpoint, simulating the conversation flow (e.g., "User:\n...\n\nAssistant:\n...\n\nAssistant:\n").
5.  **Streamlit App UI Setup:**
    *   `st.set_page_config()`: Configures the page layout.
    *   `st.markdown("<h1 ...>")`: Displays the centered title.
    *   **Session State Initialization:** Uses `if 'key' not in st.session_state:` blocks to initialize variables (`session_id`, `messages`, `uploaded_files`, `prompt1_text`, `generation_done`, `refined_output`) only once per session, preserving state across user interactions and reruns.
6.  **Sidebar UI (`with st.sidebar:`)**
    *   Provides widgets for document upload (`st.text_input` for tag, `st.file_uploader` for file).
    *   "Add Document" button (`st.button`) appends valid uploads to `st.session_state.uploaded_files`.
    *   Displays the list of currently uploaded documents.
    *   "New Chat Session" button (`st.button`) resets relevant session state variables and triggers `st.rerun()`.
7.  **Main Area - Section 1: Prompt Editor:**
    *   `st.subheader`, `st.caption`: Section titles.
    *   `st.text_area`: Displays the content of `st.session_state.prompt1_text` (loaded from `prompt_call_1.txt`) and allows the user to edit it. The edited text updates the session state variable.
8.  **Main Area - "Generate Initial Artifacts" Button Logic:**
    *   `if st.button(...)`: Contains the core logic triggered by the button.
    *   Checks if documents are uploaded.
    *   Resets session state for the new generation run.
    *   Shows a spinner (`st.spinner`).
    *   Calls `format_docs_for_prompt`.
    *   Formats the final prompt for Call 1 using the (potentially edited) `st.session_state.prompt1_text`.
    *   Calls `call_llm_service` for **LLM Call 1**.
    *   If successful, calls `parse_generator_output`.
    *   Formats the final prompt for Call 2 using `PROMPT_TEMPLATE_CALL_2_BASE` and the outputs from the previous steps.
    *   Calls `call_llm_service` for **LLM Call 2**.
    *   If successful, updates session state (`messages`, `generation_done`), shows `st.success`, and triggers `st.rerun()`.
    *   Includes error handling (`st.error`) and logging for failed LLM calls.
9.  **Main Area - Section 2: Generated Artifacts Display:**
    *   `st.subheader`: Section title.
    *   Conditional logic (`if st.session_state.generation_done or st.session_state.messages:`): Determines what to display.
    *   If generation is done, creates a scrollable container (`st.container`).
    *   Displays the first message (`st.session_state.messages[0]['content']`, which is the full output from Call 2) inside the container using `st.markdown`.
    *   If generation is *not* done, displays a placeholder caption (`st.caption`) directly under the subheader (no container).
10. **Main Area - Section 3: ChatBox:**
    *   `st.subheader`: Section title.
    *   `if st.session_state.generation_done:`: Only shows the chat interface after initial generation.
    *   Displays chat history (skipping the first message) in a scrollable container using `st.chat_message`.
    *   `st.chat_input`: Provides the text input box for the user.
    *   When the user submits input:
        *   Appends the user message to `st.session_state.messages`.
        *   Calls `format_history_for_local_llm` to create the single prompt string.
        *   Calls `call_llm_service` for the **follow-up LLM call**.
        *   Appends the assistant's response to `st.session_state.messages`.
        *   Triggers `st.rerun()` to update the displayed chat history.
    *   If generation is not done, displays a placeholder caption.
