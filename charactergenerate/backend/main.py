"""
main.py — FastAPI backend for the Character Generation app.

Endpoints:
  POST /api/load-book                 start background indexing job
  GET  /api/status                    SSE stream of job progress
  GET  /api/characters                return cached character list
  POST /api/character-details         analyze character + get RAG scenarios
  POST /api/cast-actor                cast an actor for the character
  POST /api/generate-prompt           build the final Z-Image-Turbo prompt
  POST /api/comfyui/test              verify ComfyUI is reachable
  POST /api/comfyui/generate          queue + stream SSE progress for image gen
  GET  /api/comfyui/image/{prompt_id} proxy-fetch the finished image
"""
import os
import re
import asyncio
import threading
import json
import tempfile
from typing import Any, Optional
import urllib.request
import urllib.parse
import urllib.error

import ebooklib
from ebooklib import epub
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from character_gen import (
    get_major_character_names,
    build_book_index,
    get_scenario_summaries,
    analyze_character,
    get_character_situations,
    cast_character_with_actor,
    generate_prompt_for_scenario,
)
from comfyui import (
    inject_prompt_into_workflow,
    queue_prompt,
    poll_until_done,
    get_output_image_bytes,
    test_connection,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Character Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared in-memory state (single-user dev app — no auth needed)
# ---------------------------------------------------------------------------

state: dict[str, Any] = {
    "job_status": "idle",   # idle | running | done | error
    "stage": "",
    "progress": 0,
    "total": 0,
    "message": "",
    "characters": [],
    "vectorstore": None,
    "book_title": "",
    "error": None,
}


# ---------------------------------------------------------------------------
# Background indexing thread
# ---------------------------------------------------------------------------

def load_book_task(book_path: str, book_name: str, character_source: str = "auto") -> None:
    """
    Runs in a daemon thread; writes progress into `state`.

    character_source controls how the character list is obtained:
      "wiki"        — Wikipedia only (original behaviour)
      "txt_extract" — LLM + text sampling from the local file
      "hanlp"       — HanLP MTL pipeline NER (no LLM needed)
      "auto"        — Wiki → HanLP NER → txt_extract fallback chain (default)
    """
    try:
        state["job_status"] = "running"
        state["characters"] = []
        state["vectorstore"] = None
        state["book_title"] = book_name
        state["error"] = None

        # Step 1 — Character list
        source_label = {
            "wiki":        "Wikipedia",
            "txt_extract": "local file (LLM sampling)",
            "hanlp":       "HanLP NER (MTL pipeline)",
            "auto":        "auto (Wiki → HanLP NER → local file)",
        }.get(character_source, character_source)

        state["stage"] = "characters"
        state["progress"] = 0
        state["total"] = 1
        state["message"] = f"Fetching character list via {source_label}…"

        characters = get_major_character_names(
            book_title=book_name,
            txt_filepath=book_path,
            source=character_source,
        )
        state["characters"] = characters
        state["progress"] = 1
        state["message"] = f"Found {len(characters)} characters. Building vector index…"

        # Step 2 — Vector index with live progress
        def prog_cb(stage: str, n: int, total: int, msg: str) -> None:
            state["stage"] = stage
            state["progress"] = n
            state["total"] = total
            state["message"] = msg

        vectorstore = build_book_index(book_path, progress_cb=prog_cb)
        state["vectorstore"] = vectorstore
        state["job_status"] = "done"
        state["message"] = (
            f"Ready! Indexed the book and found {len(characters)} characters."
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        state["job_status"] = "error"
        state["error"] = str(exc)
        state["message"] = f"Error: {exc}"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LoadBookRequest(BaseModel):
    book_path: str
    book_name: str
    # Controls where the character list comes from.
    # "wiki"        — Wikipedia only
    # "txt_extract" — LLM + local text sampling
    # "hanlp"       — HanLP NER MTL pipeline (no LLM needed)
    # "auto"        — Wiki → HanLP NER → txt_extract fallback chain (default)
    character_source: str = "auto"


class CharacterDetailsRequest(BaseModel):
    character_name: str


class CastActorRequest(BaseModel):
    character_name: str
    description: str
    industry: str = "hollywood"
    genre: str = ""
    decade: str = "2026"


class GeneratePromptRequest(BaseModel):
    character_name: str
    description: str
    scenario_context: str
    actor_name: str = ""
    genre: str = ""
    decade: str = "2026"
    gender: str = ""
    race: str = ""
    age: str = ""


class ComfyUITestRequest(BaseModel):
    comfy_url: str


class ComfyUIGenerateRequest(BaseModel):
    comfy_url: str
    workflow_json: str          # raw JSON string (API-format)
    prompt_text: str
    node_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/load-book")
async def load_book(req: LoadBookRequest):
    if state["job_status"] == "running":
        raise HTTPException(status_code=409, detail="A book is already being loaded.")

    req.book_path = req.book_path.strip().strip("\"'")

    if not os.path.exists(req.book_path):
        # Fallback: try relative to two levels up (project root often has data/ folder)
        alt_path = os.path.join("..", "..", req.book_path)
        if os.path.exists(alt_path):
            req.book_path = alt_path
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Book file not found. Checked: {req.book_path} and {alt_path}"
            )

    t = threading.Thread(
        target=load_book_task,
        args=(req.book_path, req.book_name, req.character_source),
        daemon=True,
    )
    t.start()
    return {"status": "started"}


def _epub_to_txt(epub_path: str, out_path: str) -> None:
    book = epub.read_epub(epub_path)
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parts.append(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))


@app.post("/api/upload-book")
async def upload_book(
    file: UploadFile = File(...),
    book_name: str = Form(...),
    character_source: str = Form("auto"),
):
    if state["job_status"] == "running":
        raise HTTPException(status_code=409, detail="A book is already being loaded.")

    filename = (file.filename or "").lower()
    if not (filename.endswith(".txt") or filename.endswith(".epub")):
        raise HTTPException(status_code=400, detail="Only .txt and .epub files are supported.")

    suffix = ".epub" if filename.endswith(".epub") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    if suffix == ".epub":
        txt_path = tmp_path.replace(".epub", ".txt")
        _epub_to_txt(tmp_path, txt_path)
        os.unlink(tmp_path)
        tmp_path = txt_path

    t = threading.Thread(
        target=load_book_task,
        args=(tmp_path, book_name, character_source),
        daemon=True,
    )
    t.start()
    return {"status": "started"}


@app.get("/api/status")
async def get_status():
    """SSE stream — the client polls this after starting a load-book job."""
    async def event_generator():
        last_snapshot: dict = {}
        while True:
            snapshot = {
                "status": state["job_status"],
                "stage": state["stage"],
                "progress": state["progress"],
                "total": state["total"],
                "message": state["message"],
                "characters": state["characters"],
            }
            if snapshot != last_snapshot:
                yield f"data: {json.dumps(snapshot)}\n\n"
                last_snapshot = dict(snapshot)

            if state["job_status"] in ("done", "error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/characters")
async def get_characters():
    return {
        "characters": state["characters"],
        "status": state["job_status"],
    }


@app.post("/api/character-details")
async def character_details(req: CharacterDetailsRequest):
    if state["vectorstore"] is None:
        raise HTTPException(
            status_code=400,
            detail="No book loaded. Call /api/load-book first.",
        )

    try:
        # Retrieve a top scene for baseline description
        situations = get_character_situations(
            vectorstore=state["vectorstore"],
            character_name=req.character_name,
            k=6,
        )
        book_text = "\n\n".join(situations)

        description = analyze_character(
            book_text=book_text,
            character_name=req.character_name,
        )

        scenarios = get_scenario_summaries(
            vectorstore=state["vectorstore"],
            character_name=req.character_name,
        )

        return {
            "description": description,
            "scenarios": scenarios,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cast-actor")
async def cast_actor(req: CastActorRequest):
    if state["vectorstore"] is None:
        raise HTTPException(
            status_code=400,
            detail="No book loaded. Call /api/load-book first.",
        )

    try:
        result = cast_character_with_actor(
            character_name=req.character_name,
            description=req.description,
            industry=req.industry,
            genre=req.genre,
            decade=req.decade,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-prompt")
async def generate_prompt(req: GeneratePromptRequest):
    if state["vectorstore"] is None:
        raise HTTPException(
            status_code=400,
            detail="No book loaded. Call /api/load-book first.",
        )

    try:
        prompt = generate_prompt_for_scenario(
            vectorstore=state["vectorstore"],
            character_name=req.character_name,
            description=req.description,
            scenario_context=req.scenario_context,
            actor_name=req.actor_name,
            genre=req.genre,
            decade=req.decade,
            gender=req.gender,
            race=req.race,
            age=req.age,
        )
        return prompt
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/comfyui/test")
async def comfyui_test(req: ComfyUITestRequest):
    """Ping the ComfyUI server to check it's up."""
    try:
        result = test_connection(req.comfy_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/comfyui/generate")
async def comfyui_generate(req: ComfyUIGenerateRequest):
    """
    Inject prompt, queue it in ComfyUI, and stream progress via SSE.
    """
    try:
        workflow = inject_prompt_into_workflow(
            workflow_json=req.workflow_json,
            prompt_text=req.prompt_text,
            node_id=req.node_id,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Workflow error: {e}")

    try:
        prompt_id = queue_prompt(req.comfy_url, workflow)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ComfyUI queue error: {e}")

    async def stream():
        try:
            for event in poll_until_done(req.comfy_url, prompt_id):
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/comfyui/image/{prompt_id}")
async def get_comfyui_image(prompt_id: str, comfy_url: str):
    """Proxy-fetch the finished image from ComfyUI."""
    try:
        img_bytes = get_output_image_bytes(comfy_url, prompt_id)
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
