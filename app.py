import tempfile
from fastapi import FastAPI, File, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
import requests
import json
import os
app = FastAPI()

# Define the input structure for questions
class Questions(BaseModel):
    questions: list[str]

# Load document based on file type (PDF or JSON)
def load_document(file: UploadFile):
    if file.filename.endswith(".pdf"):
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        
        # Load the PDF using PyPDFLoader with the temporary file path
        loader = PyPDFLoader(tmp_path)
        document = loader.load()
        text = " ".join([page.page_content for page in document])
        
        # Delete the temporary file after processing
        os.remove(tmp_path)
    
    elif file.filename.endswith(".json"):
        content = json.load(file.file)
        text = " ".join([entry["text"] for entry in content])
    else:
        raise ValueError("Unsupported file type")
    
    return text

# Use Ollama streaming API to query the model
def query_ollama_stream(prompt, model="tinyllama"):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response_content = ""
    with requests.post(url, data=json.dumps(payload), headers=headers, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                response_content += data['message']['content']
    
    return response_content

# API endpoint to process question answering
@app.post("/answer")
async def answer_questions(questions_file: UploadFile, document_file: UploadFile):
    # Load the questions from the JSON file
    questions_data = await questions_file.read()
    questions = json.loads(questions_data)["questions"]

    # Load the document content from the uploaded file
    document_text = load_document(document_file)
    
    # Use Ollama TinyLlama model to answer the questions
    responses = []
    for question in questions:
        prompt = f"Answer the question based on the following document:\nDocument: {document_text}\nQuestion: {question}"
        answer = query_ollama_stream(prompt, model="tinyllama")
        responses.append({"question": question, "answer": answer})
    
    return responses

# Run the API using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
