# --- Import necessary libraries ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pdfminer.high_level import extract_text
import docx2txt
from PIL import Image
from typing import List, Dict, Any, Optional
import base64
import re
import io
import os
from enum import Enum
from collections import defaultdict

from pydantic import BaseModel, Field, ValidationError # ValidationError for specific handling

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn

# --- Load Environment Variables ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Pydantic Models ---
class ContextLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Internal model to hold all processed information for a document
class ProcessedDocument(BaseModel):
    identifier: str = Field(..., description="A unique identifier for the document, derived from original filename (sequence added for duplicates).")
    original_filename: str
    context_level: ContextLevel # This remains non-optional as a default will be applied
    user_provided_metadata_string: Optional[str] = Field(None, description="Arbitrary string metadata provided by the user.")
    type: str  # "text", "image", or "error"
    content: str # Extracted text, base64 data URI, or error message

# MODIFIED: Pydantic models for the structured request payload
class FileContextMetadata(BaseModel):
    # context_level is now optional and defaults to MEDIUM if not provided
    context_level: Optional[ContextLevel] = Field(default=ContextLevel.MEDIUM, description="Optional context level for this file. Defaults to 'medium' if not provided.")
    document_metadata: Optional[str] = Field(None, description="Arbitrary string metadata provided by the user for this specific file.")

class APIPayload(BaseModel):
    metadata: List[FileContextMetadata] = Field([], description="Metadata for each uploaded file, in order corresponding to the files list.")


# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions (extract_txt, extract_pdf, extract_docx, extract_img_data_uri) ---
# These functions remain UNCHANGED.
def extract_txt(file_content: bytes, document_identifier: str) -> str:
    try:
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            text = file_content.decode("latin-1")
        return text.strip()
    except Exception as e:
        print(f"Error reading text file with identifier '{document_identifier}': {e}")
        return f"Error reading file with identifier '{document_identifier}': {e}"

def extract_pdf(file_content: bytes, document_identifier: str) -> str:
    try:
        pdf_stream = io.BytesIO(file_content)
        text = extract_text(pdf_stream)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error reading PDF file with identifier '{document_identifier}': {e}")
        return f"Error reading file with identifier '{document_identifier}': {e}"

def extract_docx(file_content: bytes, document_identifier: str) -> str:
    try:
        docx_stream = io.BytesIO(file_content)
        text = docx2txt.process(docx_stream)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error reading DOCX file with identifier '{document_identifier}': {e}")
        return f"Error reading file with identifier '{document_identifier}': {e}"

def extract_img_data_uri(file_content: bytes, document_identifier: str) -> Optional[str]:
    try:
        with Image.open(io.BytesIO(file_content)) as img:
            img_format = img.format or 'jpeg'
        content_type = f"image/{img_format.lower()}"
        base64_image = base64.b64encode(file_content).decode("utf-8")
        return f"data:{content_type};base64,{base64_image}"
    except Exception as e:
        print(f"Error processing image file with identifier '{document_identifier}': {e}")
        return None

# --- Prompt Generation and LLM Interaction Functions ---
# generate_prompt and generate_llm_response remain UNCHANGED.
def generate_prompt(developer_instructions: str, processed_documents: List[ProcessedDocument]) -> str:
    context_str = "**BEGIN TASK CONTEXT**\n\n"
    def sort_key(doc: ProcessedDocument):
        level_order = {ContextLevel.HIGH: 0, ContextLevel.MEDIUM: 1, ContextLevel.LOW: 2}
        type_order = {"text": 0, "image": 1, "error": 2}
        # doc.context_level will always be a valid ContextLevel enum due to Pydantic default
        return (level_order.get(doc.context_level, 3), # Default to 3 if somehow not in map (should not happen)
                type_order.get(doc.type, 3),
                doc.identifier)
    sorted_documents = sorted(processed_documents, key=sort_key)
    text_documents_parts = []
    image_documents_parts = []
    error_documents_parts = []
    for doc in sorted_documents:
        doc_header = f"Document Identifier: {doc.identifier}\n"
        if doc.identifier != doc.original_filename:
            doc_header += f"Original Filename: {doc.original_filename}\n"
        doc_header += f"Context Level: {doc.context_level.value}\n" # .value will work as it's always a ContextLevel
        if doc.user_provided_metadata_string:
            doc_header += f"User Provided Metadata: {doc.user_provided_metadata_string}\n"
        if doc.type == "text":
            text_documents_parts.append(
                f"--- Start of content from: {doc.identifier} (Context: {doc.context_level.value}) ---\n"
                f"{doc_header}\n"
                f"Content:\n{doc.content}\n"
                f"--- End of content from: {doc.identifier} ---\n\n"
            )
        elif doc.type == "image":
            image_info = f"- Image Identifier: {doc.identifier} (Context: {doc.context_level.value})\n"
            if doc.identifier != doc.original_filename:
                image_info += f"  Original Filename: {doc.original_filename}\n"
            if doc.user_provided_metadata_string:
                image_info += f"  User Provided Metadata: {doc.user_provided_metadata_string}\n"
            image_documents_parts.append(image_info)
        elif doc.type == "error":
            error_documents_parts.append(
                f"- Document Identifier: {doc.identifier} (Context: {doc.context_level.value})\n"
                f"  Original Filename: {doc.original_filename}\n"
                f"  Processing Status: {doc.content}\n\n"
            )
    if text_documents_parts:
        context_str += "### Provided Text Documents (sorted by importance):\n\n"
        context_str += "".join(text_documents_parts)
    else:
        context_str += "### Provided Text Documents:\nNo text documents were provided or successfully processed.\n\n"
    if image_documents_parts:
        context_str += "### Provided Image Files (sorted by importance):\n"
        context_str += "".join(image_documents_parts)
        context_str += "(Note: The visual content of these images is provided separately to the model if they were processed successfully.)\n\n"
    else:
        context_str += "### Provided Image Files:\nNo image files were provided or successfully processed.\n\n"
    if error_documents_parts:
        context_str += "### Documents with Processing Issues:\n"
        context_str += "".join(error_documents_parts)
    context_str += "**END TASK CONTEXT**\n\n"
    prompt = f"""**Role:** You are an AI assistant specialized in software development tasks, particularly UI/UX design and wireframing. Your capabilities include analyzing requirements, translating features into visual layouts, and describing user interface elements and interactions.
**Objective:** Based *strictly* on the provided context (Text Documents, Image Files, and their associated metadata, particularly their 'Identifier' - which is the original filename with a sequence number if duplicates exist - and 'Context Level', listed above, if any) and the specific 'Instructions' below, generate a detailed wireframe description. Synthesize information accurately from all provided sources to outline the structure, key elements, and basic user flow for the requested interface. Consider the 'Context Level' of each document to understand its relative importance.
{context_str}
### Instructions (Your Task)
{developer_instructions}
*Focus on describing the visual layout, placement of key UI elements (buttons, input fields, navigation, content areas), and essential user interactions for the wireframe.*
### Output Requirements
- Generate *only* the detailed wireframe description requested in the 'Instructions'.
- Ensure the output directly addresses the instructions and accurately incorporates relevant details from the provided context, respecting the context level and using document identifiers (original filenames with sequence numbers for duplicates) for reference.
- Format the output clearly, often using structured text or markdown lists to describe each screen or component of the wireframe. Detail the layout, elements present, and their basic functions or interactions (e.g., "Button labeled 'Submit' navigates to the confirmation screen").
- Do not add conversational introductions, conclusions, or explanations unless the instructions explicitly ask for them. Focus solely on generating the requested wireframe description.
"""
    return prompt.strip()

def generate_llm_response(prompt: str, processed_documents: List[ProcessedDocument]) -> str:
    if not google_api_key:
        raise HTTPException(status_code=500, detail="Server configuration error: Google API Key not set.")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.35,
            max_tokens=30000,
            google_api_key=google_api_key
        )
        content_for_llm: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for doc in processed_documents:
            if doc.type == "image" and doc.content.startswith("data:image"):
                content_for_llm.append({
                    "type": "image_url",
                    "image_url": {"url": doc.content}
                })
                print(f"Added image with identifier '{doc.identifier}' to LLM input.")
        message = HumanMessage(content=content_for_llm)
        print("Invoking LLM...")
        response = llm.invoke([message])
        print("LLM invocation complete.")
        return response.content
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response from AI model: {e}")


# --- FastAPI Endpoint (Docstring Updated) ---
@app.post("/wireframe")
async def process_instruction(
    developer_instructions: str = Form(..., description="Mandatory instructions for the task."),
    payload_json: Optional[str] = Form(None, description="An optional JSON string containing an ordered list 'metadata' for each uploaded file (if any). Each entry can have 'context_level' (optional, defaults to 'medium') and 'document_metadata' (optional). If omitted, files will default to 'medium' context_level and no specific document_metadata."),
    files: List[UploadFile] = File([], description="Optional list of files. If 'payload_json' is provided and contains a 'metadata' list, it must have a corresponding entry for each file, in the same order.")
):
    """
    Processes developer instructions with an optional list of context files.

    - "developer_instructions": A mandatory form field for the main task instructions.
    - "payload_json": (Optional) A form field for a JSON string. If provided, it must contain a JSON object
                      with a single key "metadata". "metadata" should be a list of objects, where each object can have:
                        - "context_level": (optional string: "low", "medium", or "high"; defaults to "medium" if omitted for a file)
                        - "document_metadata": (optional string) user-provided simple text metadata for the file.
                      If "payload_json" and its "metadata" list are provided, this list must be in the same order
                      as the uploaded 'files'. If "payload_json" is provided but "metadata" is empty (e.g., '{"metadata": []}')
                      and files are uploaded, an error will occur.
                      If "payload_json" is omitted entirely and files are uploaded, all files will default to a "medium"
                      context_level and will have no specific "document_metadata".
    - "files": (Optional) An optional list of uploaded files.

    A unique identifier will be generated for each file based on its original filename,
    with sequence numbers added for duplicate filenames within the same request.
    """
    file_specific_metadata_list: List[FileContextMetadata] = []
    api_payload_data: Optional[APIPayload] = None

    if payload_json:
        try:
            api_payload_data = APIPayload.model_validate_json(payload_json)
            file_specific_metadata_list = api_payload_data.metadata
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"Invalid 'payload_json': {e.errors()}")

    all_processed_documents: List[ProcessedDocument] = []
    num_files = len(files)
    num_metadata_entries = len(file_specific_metadata_list)

    print(f"Received request with {num_files} file(s). Developer instructions: '{developer_instructions[:100]}...'")
    if payload_json:
        print(f"Payload JSON provided. Number of metadata entries: {num_metadata_entries}")
    else:
        print("No payload_json provided. Files will use default metadata settings.")


    # Validation: Only if payload_json was provided, then metadata count must match file count (if files exist)
    # or metadata should not be provided if no files exist.
    if payload_json: # These checks apply only if payload_json was explicitly sent
        if num_files > 0 and num_metadata_entries != num_files:
            raise HTTPException(
                status_code=400,
                detail=f"If 'payload_json' is provided and files are uploaded, the 'metadata' list in 'payload_json' must have the same number of entries as files. Received {num_files} files and {num_metadata_entries} metadata entries."
            )
        if num_files == 0 and num_metadata_entries > 0:
            raise HTTPException(
                status_code=400,
                detail="'metadata' entries were provided in 'payload_json', but no files were uploaded. If providing 'payload_json' with no files, its 'metadata' list should be empty (i.e., '{\"metadata\": []}')."
            )

    filename_counts: Dict[str, int] = defaultdict(int)

    try:
        for i, file_upload in enumerate(files):
            original_filename = file_upload.filename or f"unknown_file_{i}"

            filename_counts[original_filename] += 1
            if filename_counts[original_filename] == 1:
                current_doc_identifier = original_filename
            else:
                current_doc_identifier = f"{original_filename} ({filename_counts[original_filename] - 1})"

            current_context_level: ContextLevel
            current_user_provided_string: Optional[str]

            if payload_json and i < num_metadata_entries:
                # payload_json was provided and there's a corresponding metadata entry
                doc_meta_entry = file_specific_metadata_list[i]
                # Pydantic default for context_level (MEDIUM) is applied if not in JSON
                current_context_level = doc_meta_entry.context_level
                current_user_provided_string = doc_meta_entry.document_metadata
            else:
                # No payload_json, or payload_json provided but not enough metadata entries (e.g. {"metadata":[]})
                # For this file, apply global defaults.
                current_context_level = ContextLevel.MEDIUM
                current_user_provided_string = None


            # Fallback, though Pydantic default on FileContextMetadata should prevent current_context_level from being None
            # if doc_meta_entry was valid. This primarily handles the 'else' case above more explicitly.
            if current_context_level is None:
                current_context_level = ContextLevel.MEDIUM


            print(f"Processing file: Identifier='{current_doc_identifier}', Original Filename='{original_filename}', Context: {current_context_level.value}, User Metadata: '{current_user_provided_string}', Content-Type: {file_upload.content_type}")

            file_content = await file_upload.read()
            content_type_from_upload = file_upload.content_type
            file_ext = Path(original_filename).suffix.lower()

            processed_content: str = ""
            doc_type: str = "error"

            if content_type_from_upload and content_type_from_upload.startswith('image/'):
                uri = extract_img_data_uri(file_content, current_doc_identifier)
                if uri:
                    processed_content = uri
                    doc_type = "image"
                else:
                    processed_content = f"Error processing image file '{current_doc_identifier}'. Check server logs."
            elif file_ext == '.pdf' or content_type_from_upload == 'application/pdf':
                processed_content = extract_pdf(file_content, current_doc_identifier)
                doc_type = "text" if not processed_content.startswith("Error reading file") else "error"
            elif file_ext == '.docx' or content_type_from_upload == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                processed_content = extract_docx(file_content, current_doc_identifier)
                doc_type = "text" if not processed_content.startswith("Error reading file") else "error"
            elif file_ext == '.txt' or content_type_from_upload == 'text/plain':
                processed_content = extract_txt(file_content, current_doc_identifier)
                doc_type = "text" if not processed_content.startswith("Error reading file") else "error"
            else:
                processed_content = f"Unsupported file type for '{current_doc_identifier}' (type: '{content_type_from_upload}' / ext: '{file_ext}')."
                print(processed_content)
                doc_type = "error"

            all_processed_documents.append(
                ProcessedDocument(
                    identifier=current_doc_identifier,
                    original_filename=original_filename,
                    context_level=current_context_level, # This will have a valid ContextLevel enum value
                    user_provided_metadata_string=current_user_provided_string,
                    type=doc_type,
                    content=processed_content
                )
            )

        print("Generating prompt...")
        prompt = generate_prompt(developer_instructions, all_processed_documents)
        # print(prompt) # Optionally print the full prompt for debugging
        print("Generating LLM response...")
        response_content = generate_llm_response(prompt, all_processed_documents)

        return {"response": response_content}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred in /wireframe endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        for file_upload in files:
            if hasattr(file_upload, 'file') and hasattr(file_upload.file, 'closed') and not file_upload.file.closed:
                try:
                    await file_upload.close()
                except Exception:
                    pass


# --- Run the App (for local development) ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
