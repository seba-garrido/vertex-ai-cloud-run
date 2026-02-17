from fastapi import FastAPI
from pydantic import BaseModel
import os
from typing import TypedDict, Literal

import vertexai
from vertexai.generative_models import GenerativeModel

from langgraph.graph import StateGraph, END

app = FastAPI(title="Vertex AI Cloud Run + LangGraph")

class PromptIn(BaseModel):
    prompt: str

@app.get("/health")
def health():
    return {"status": "ok"}

def _get_vertex_model() -> GenerativeModel:
    project = os.environ["VERTEX_PROJECT_ID"]
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    model_name = os.environ.get("VERTEX_MODEL", "gemini-1.5-pro")

    vertexai.init(project=project, location=location)
    return GenerativeModel(model_name)

# ---- Tu endpoint original (lo dejamos igual, solo reusamos helper) ----
@app.post("/generate")
def generate(body: PromptIn):
    model = _get_vertex_model()
    resp = model.generate_content(body.prompt)
    return {"model": model._model_name if hasattr(model, "_model_name") else "configured", "output": resp.text}

# ---- LangGraph POC ----
class GraphState(TypedDict):
    prompt: str
    route: Literal["llm"]
    output: str

def route_node(state: GraphState) -> GraphState:
    # POC: siempre va al LLM (en el futuro acá decides rutas)
    return {**state, "route": "llm"}

def llm_node(state: GraphState) -> GraphState:
    model = _get_vertex_model()
    resp = model.generate_content(state["prompt"])
    return {**state, "output": resp.text}

def format_node(state: GraphState) -> GraphState:
    # POC: devuelve tal cual; acá podrías limpiar/estructurar
    return state

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("route", route_node)
    g.add_node("llm", llm_node)
    g.add_node("format", format_node)

    g.set_entry_point("route")
    g.add_edge("route", "llm")
    g.add_edge("llm", "format")
    g.add_edge("format", END)

    return g.compile()

graph = build_graph()

@app.post("/langgraph-demo")
def langgraph_demo(body: PromptIn):
    result = graph.invoke({"prompt": body.prompt, "route": "llm", "output": ""})
    return {"output": result["output"]}
