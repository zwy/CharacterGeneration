import os
import threading
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from character_gen import (
    get_major_character_names,
    build_book_index,
    get_character_description,
    get_character_scenarios,
    generate_image_prompt,
    LLMConfig,
)

# ---------------------------------------------------------------------------
# Shared state (single-user, in-memory)
# ---------------------------------------------------------------------------
state: dict = {
    "status":      "idle",   # idle | loading | done | error
    "stage":       "",
    "progress":    0,
    "total":       0,
    "message":     "",
    "characters":  [],
    "vectorstore": None,
    "book_title":  "",
    "error":       None,
}


# ---------------------------------------------------------------------------
# Background task — load + index a book
# ---------------------------------------------------------------------------
def load_book_task(book_path: str, book_name: str, character_source: str = "auto") -> None:
    """
    Runs in a background thread.
    character_source controls how the character list is obtained:
      - "wiki"        : Wikipedia lookup
      - "txt_extract" : LLM-based sampling from the local file
      - "hanlp"       : HanLP NER MTL pipeline (no LLM needed)
      - "auto"        : wiki → hanlp → txt_extract fallback chain
    """
    global state
    try:
        state["status"]     = "loading"
        state["characters"] = []
        state["vectorstore"] = None
        state["book_title"] = book_name
        state["error"]      = None

        # Step 1 — Character list
        source_label = {
            "wiki":        "Wikipedia",
            "txt_extract": "local file (LLM sampling)",
            "hanlp":       "HanLP NER (MTL pipeline)",
            "auto":        "auto (Wiki → HanLP NER → local file)",
        }.get(character_source, character_source)

        state["stage"]    = "characters"
        state["progress"] = 0
        state["total"]    = 1
        state["message"]  = f"Fetching character list via {source_label}…"

        characters = get_major_character_names(
            book_title=book_name,
            txt_filepath=book_path,
            source=character_source,
        )
        state["characters"] = characters
        state["progress"]   = 1
        state["message"]    = f"Found {len(characters)} characters. Building vector index…"

        # Step 2 — Vector index with live progress
        def prog_cb(stage: str, n: int, total: int, msg: str) -> None:
            state["stage"]    = stage
            state["progress"] = n
            state["total"]    = total
            state["message"]  = msg

        vectorstore = build_book_index(book_path, progress_cb=prog_cb)
        state["vectorstore"] = vectorstore
        state["status"]      = "done"
        state["message"]     = "Book loaded and indexed successfully."

    except Exception as exc:
        state["status"]  = "error"
        state["error"]   = str(exc)
        state["message"] = str(exc)
    finally:
        # Clean up temp upload file if it lives in /tmp
        if book_path.startswith("/tmp") or book_path.startswith(os.sep + "tmp"):
            try:
                os.remove(book_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LoadBookRequest(BaseModel):
    book_path: str
    book_name: str
    character_source: str = "auto"


class DescribeRequest(BaseModel):
    book_title: str
    character_name: str


class ScenariosRequest(BaseModel):
    book_title: str
    character_name: str
    character_description: str


class PromptRequest(BaseModel):
    book_title: str
    character_name: str
    character_description: str
    scenario_context: str
    genre: str = "Cinematic Realism"
    decade: str = "Present Day"
    actor_name: str = ""
    gender: str = ""
    race: str = ""
    age: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/load-book")
def load_book(req: LoadBookRequest, background_tasks: BackgroundTasks):
    """Start indexing from a server-side file path."""
    if state["status"] == "loading":
        raise HTTPException(status_code=409, detail="A book is already being loaded.")
    state["status"] = "loading"
    background_tasks.add_task(
        load_book_task, req.book_path, req.book_name, req.character_source
    )
    return {"message": "Loading started"}


@app.post("/api/upload-book")
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    book_name: str = Form(...),
    character_source: str = Form("auto"),
):
    """Accept a file upload, save to /tmp, then kick off the background job."""
    if state["status"] == "loading":
        raise HTTPException(status_code=409, detail="A book is already being loaded.")

    suffix = Path(file.filename).suffix
    tmp_path = f"/tmp/{book_name}{suffix}"
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    state["status"] = "loading"
    background_tasks.add_task(
        load_book_task, tmp_path, book_name, character_source
    )
    return {"message": "Upload received, loading started"}


@app.get("/api/status")
def stream_status():
    """SSE endpoint — push state updates to the client."""
    def event_stream():
        last_snapshot = None
        while True:
            snapshot = {
                "status":   state["status"],
                "stage":    state["stage"],
                "progress": state["progress"],
                "total":    state["total"],
                "message":  state["message"],
            }
            if state["status"] == "done":
                snapshot["characters"] = state["characters"]

            if snapshot != last_snapshot:
                yield f"data: {json.dumps(snapshot)}\n\n"
                last_snapshot = snapshot

            if state["status"] in ("done", "error"):
                break
            time.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/describe")
def describe(req: DescribeRequest):
    if state["vectorstore"] is None:
        raise HTTPException(status_code=400, detail="No book loaded.")
    description = get_character_description(
        vectorstore=state["vectorstore"],
        book_title=req.book_title,
        character_name=req.character_name,
    )
    return {"description": description}


@app.post("/api/scenarios")
def scenarios(req: ScenariosRequest):
    if state["vectorstore"] is None:
        raise HTTPException(status_code=400, detail="No book loaded.")
    result = get_character_scenarios(
        vectorstore=state["vectorstore"],
        book_title=req.book_title,
        character_name=req.character_name,
        character_description=req.character_description,
    )
    return {"scenarios": result}


@app.post("/api/generate-prompt")
def generate_prompt(req: PromptRequest):
    if state["vectorstore"] is None:
        raise HTTPException(status_code=400, detail="No book loaded.")
    prompt = generate_image_prompt(
        vectorstore=state["vectorstore"],
        book_title=req.book_title,
        character_name=req.character_name,
        character_description=req.character_description,
        scenario_context=req.scenario_context,
        genre=req.genre,
        decade=req.decade,
        actor_name=req.actor_name,
        gender=req.gender,
        race=req.race,
        age=req.age,
    )
    return {"prompt": prompt}


@app.get("/api/characters")
def get_characters():
    return {"characters": state["characters"]}


# ---------------------------------------------------------------------------
# Serve React frontend (production build)
# ---------------------------------------------------------------------------
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
