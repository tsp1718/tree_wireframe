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
from collections import defaultdict

from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn

# --- Load Environment Variables ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Pydantic Models for /wireframe endpoint ---
class ProcessedDocument(BaseModel):
    identifier: str = Field(..., description="A unique identifier for the document, derived from original filename (sequence added for duplicates).")
    original_filename: str
    type: str  # "text", "image", or "error"
    content: str # Extracted text, base64 data URI, or error message


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multimodal Wireframe AI Assistant",
    description="An AI assistant that generates wireframe instructions based on user descriptions and/or uploaded documents."
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions for /wireframe endpoint ---
def extract_txt(file_content: bytes, document_identifier: str) -> str:
    try:
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            text = file_content.decode("latin-1")
        return text.strip()
    except Exception as e:
        return f"Error reading file with identifier '{document_identifier}': {e}"

def extract_pdf(file_content: bytes, document_identifier: str) -> str:
    try:
        pdf_stream = io.BytesIO(file_content)
        text = extract_text(pdf_stream)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        return f"Error reading file with identifier '{document_identifier}': {e}"

def extract_docx(file_content: bytes, document_identifier: str) -> str:
    try:
        docx_stream = io.BytesIO(file_content)
        text = docx2txt.process(docx_stream)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
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

# --- Prompt Generation and LLM Functions ---

# Prompt generator for the /wireframe endpoint
def generate_wireframe_prompt(developer_instructions: str, processed_documents: List[ProcessedDocument]) -> str:
    context_str = "**BEGIN TASK CONTEXT**\n\n"
    sorted_documents = sorted(processed_documents, key=lambda doc: ({"text": 0, "image": 1, "error": 2}.get(doc.type, 3), doc.identifier))
    
    text_documents_parts, image_documents_parts, error_documents_parts = [], [], []

    for doc in sorted_documents:
        doc_header = f"Document Identifier: {doc.identifier}\n"
        if doc.identifier != doc.original_filename:
            doc_header += f"Original Filename: {doc.original_filename}\n"
        
        if doc.type == "text":
            text_documents_parts.append(f"--- Start of content from: {doc.identifier} ---\n{doc_header}\nContent:\n{doc.content}\n--- End of content from: {doc.identifier} ---\n\n")
        elif doc.type == "image":
            image_documents_parts.append(f"- Image Identifier: {doc.identifier}\n" + (f"  Original Filename: {doc.original_filename}\n" if doc.identifier != doc.original_filename else ""))
        elif doc.type == "error":
            error_documents_parts.append(f"- Document Identifier: {doc.identifier}\n  Original Filename: {doc.original_filename}\n  Processing Status: {doc.content}\n\n")

    context_str += "### Provided Text Documents:\n" + ("".join(text_documents_parts) if text_documents_parts else "No text documents were provided or successfully processed.\n\n")
    context_str += "### Provided Image Files:\n" + ("".join(image_documents_parts) + "(Note: The visual content of these images is provided separately to the model.)\n\n" if image_documents_parts else "No image files were provided or successfully processed.\n\n")
    if error_documents_parts:
        context_str += "### Documents with Processing Issues:\n" + "".join(error_documents_parts)
    context_str += "**END TASK CONTEXT**\n\n"

    return f"""**Role:** You are an AI assistant specialized in software development tasks, particularly UI/UX design and wireframing.
**Objective:** Based *strictly* on the provided context (Text Documents, Image Files) and the 'Instructions' below, generate a detailed wireframe description.
{context_str}
### Instructions (Your Task)
{developer_instructions}
*Focus on describing the visual layout, placement of key UI elements, and essential user interactions for the wireframe.*
### Output Requirements
- Generate *only* the detailed wireframe description requested.
- Ensure the output directly addresses the instructions and accurately incorporates relevant details from the provided context.
- Format the output clearly, using structured text or markdown lists.
- Do not add conversational introductions or conclusions.
""".strip()

# Prompt generator for the /instruct endpoint
def generate_instruct_prompt(description: str) -> str:
    return f"""**Role:** You are an AI assistant specialized in software development tasks. Your goal is to generate wireframe instructions based on the provided description.

### Instructions (Your Task)
Based on the following description, provide comprehensive wireframe instructions that cover:

- **Screen Title or Page Name**: A clear title for the UI screen/page.
- **Key UI Elements**: A detailed list of all forms, buttons, input fields, menus, widgets, and other interactive or static UI components.
- **UI Layout Guidance**: Specific instructions on alignment, positioning, and the use of containers.
- **Primary User Flow**: A step-by-step narrative describing user interaction.
- **Responsive Behavior**: Notes on how the UI should adapt to different screen sizes.
- **Visual or UX Notes**: Suggestions for color schemes, error states, icons, etc.

**Description:**
{description}

*Example for inspiration:*
> *'The wireframe for the 'Login Screen' should include a centered screen title ('Sign In'), two input fields ('Email', 'Password'), a primary button ('Login'), and a 'Forgot Password?' link. Display inline error messages. Layout must be responsive. Consider soft shadows and rounded corners for inputs.'*
""".strip()

# General LLM interaction function
def generate_llm_response(prompt: str, processed_documents: Optional[List[ProcessedDocument]] = None) -> str:
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
        if processed_documents:
            for doc in processed_documents:
                if doc.type == "image" and doc.content.startswith("data:image"):
                    content_for_llm.append({"type": "image_url", "image_url": {"url": doc.content}})
        
        message = HumanMessage(content=content_for_llm)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response from AI model: {e}")

# --- FastAPI Endpoints ---

@app.post("/wireframe", summary="Generate wireframe from instructions and documents")
async def process_wireframe_request(
    developer_instructions: str = Form(..., description="Mandatory instructions for the task."),
    files: List[UploadFile] = File([], description="Optional list of files to provide context.")
):
    """
    Processes developer instructions and an optional list of context files to generate a wireframe description.
    """
    all_processed_documents: List[ProcessedDocument] = []
    filename_counts: Dict[str, int] = defaultdict(int)

    try:
        for i, file_upload in enumerate(files):
            original_filename = file_upload.filename or f"unknown_file_{i}"
            filename_counts[original_filename] += 1
            current_doc_identifier = f"{original_filename} ({filename_counts[original_filename] - 1})" if filename_counts[original_filename] > 1 else original_filename

            file_content = await file_upload.read()
            content_type = file_upload.content_type
            file_ext = Path(original_filename).suffix.lower()

            doc_type, processed_content = "error", ""

            if content_type and content_type.startswith('image/'):
                uri = extract_img_data_uri(file_content, current_doc_identifier)
                doc_type, processed_content = ("image", uri) if uri else ("error", f"Error processing image file '{current_doc_identifier}'.")
            elif file_ext == '.pdf' or content_type == 'application/pdf':
                text = extract_pdf(file_content, current_doc_identifier)
                doc_type, processed_content = ("text", text) if not text.startswith("Error") else ("error", text)
            elif file_ext == '.docx' or content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                text = extract_docx(file_content, current_doc_identifier)
                doc_type, processed_content = ("text", text) if not text.startswith("Error") else ("error", text)
            elif file_ext == '.txt' or content_type == 'text/plain':
                text = extract_txt(file_content, current_doc_identifier)
                doc_type, processed_content = ("text", text) if not text.startswith("Error") else ("error", text)
            else:
                processed_content = f"Unsupported file type for '{current_doc_identifier}'."

            all_processed_documents.append(ProcessedDocument(identifier=current_doc_identifier, original_filename=original_filename, type=doc_type, content=processed_content))

        prompt = generate_wireframe_prompt(developer_instructions, all_processed_documents)
        response_content = generate_llm_response(prompt, all_processed_documents)
        return {"response": response_content}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        for file_upload in files:
            await file_upload.close()

@app.post("/instruct", summary="Generate wireframe instructions from a description")
async def process_instruction(
    description: str = Form(..., description="Mandatory instructions for the wireframe generation task.")
) -> Dict[str, str]:
    """
    Receives a description and generates detailed wireframe instructions using an AI model.
    """
    try:
        prompt = generate_instruct_prompt(description)
        response_content = generate_llm_response(prompt)
        return {"response": response_content}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Run the App (for local development) ---
if __name__ == "__main__":
    import traceback
    print("Starting FastAPI server...")
    print("Access the API documentation at: http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
