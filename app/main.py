from fastapi import FastAPI
from pydantic import BaseModel
import os

import vertexai
from vertexai.generative_models import GenerativeModel

app = FastAPI(title="Vertex AI Cloud Run")

class PromptIn(BaseModel):
    prompt: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(body: PromptIn):
    project = os.environ["VERTEX_PROJECT_ID"]
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    model_name = os.environ.get("VERTEX_MODEL", "gemini-1.5-pro")

    vertexai.init(project=project, location=location)
    model = GenerativeModel(model_name)
    resp = model.generate_content(body.prompt)

    return {"model": model_name, "output": resp.text}
