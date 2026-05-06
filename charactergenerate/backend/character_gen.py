"""
character_gen.py
Core logic for the Character Generation app.
Refactored from CharacterImageGeneration.py with:
  - progress callbacks for SSE streaming
  - get_scenario_summaries() — RAG-derived scenario dropdown
  - generate_prompt_for_scenario() — single call for one scene
  - get_major_character_names_from_txt() — LLM+sampling fallback when Wiki is unavailable
"""

import os
import json
import urllib.request
import urllib.parse

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load .env file BEFORE reading any environment variables
load_dotenv()

OLLAMA_MODEL = "Gemma4E4B"
OLLAMA_HOST = "http://localhost:11434"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "ollama")
LLM_MODEL    = os.getenv("LLM_MODEL",    "Gemma4E4B:latest")

# ---------------------------------------------------------------------------
# Character source strategy setting
# "wiki"        — fetch from Wikipedia (original behaviour, good for well-known books)
# "txt_extract" — sample the TXT file and use LLM to extract names (good for local/CN novels)
# "auto"        — try Wiki first; if it returns 0 results, fall back to txt_extract
# ---------------------------------------------------------------------------
CHARACTER_SOURCE = os.getenv("CHARACTER_SOURCE", "auto")


def get_llm_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


# ---------------------------------------------------------------------------
# 1a. Wikipedia — major character names (original)
# ---------------------------------------------------------------------------

def get_major_character_names_from_wiki(book_title: str) -> list[str]:
    """Query Wikipedia for the book and extract major character names via LLM."""
    try:
        search_query = urllib.parse.quote(book_title)
        search_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={search_query}&utf8=&format=json"
        )
        req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            search_data = json.loads(response.read().decode("utf-8"))

        if not search_data["query"]["search"]:
            return []

        page_title = search_data["query"]["search"][0]["title"]
        content_query = urllib.parse.quote(page_title)
        content_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&explaintext=1&titles={content_query}&format=json"
        )
        req2 = urllib.request.Request(content_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2) as response:
            content_data = json.loads(response.read().decode("utf-8"))

        pages = content_data["query"]["pages"]
        page_id = list(pages.keys())[0]
        content = pages[page_id].get("extract", "")
        if not content:
            return []

        client = get_llm_client()
        prompt = (
            f"Extract a comma-separated list of major character names from the following "
            f"Wikipedia article about the book '{book_title}'. "
            f"Return ONLY the comma-separated list of names, no other text.\n\n"
            f"Article text:\n{content[:15000]}"
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        text_response = response.choices[0].message.content
        major_characters = [n.strip() for n in text_response.split(",") if n.strip()]
        return major_characters

    except Exception as e:
        print(f"[Wikipedia] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# 1b. TXT sampling — major character names (new LLM+sampling strategy)
# ---------------------------------------------------------------------------

def _sample_txt(txt_filepath: str, total_chars: int = 15000) -> str:
    """
    Sample text from beginning, middle, and end of the file.
    Returns a combined excerpt of roughly `total_chars` characters.
    Handles UTF-8 and GBK encodings automatically.
    """
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            with open(txt_filepath, "r", encoding=enc, errors="ignore") as f:
                full_text = f.read()
            break
        except Exception:
            continue
    else:
        return ""

    length = len(full_text)
    if length <= total_chars:
        return full_text

    chunk = total_chars // 3
    head   = full_text[:chunk]
    mid_start = (length // 2) - (chunk // 2)
    middle = full_text[mid_start: mid_start + chunk]
    tail   = full_text[length - chunk:]

    return (
        "=== Beginning ===\n" + head +
        "\n\n=== Middle ===\n" + middle +
        "\n\n=== End ===\n" + tail
    )


def get_major_character_names_from_txt(txt_filepath: str, book_title: str = "") -> list[str]:
    """
    Sample the TXT file from head / middle / tail, then ask the LLM to
    extract major character names.  Works for Chinese web novels and any
    local file that has no Wikipedia entry.
    """
    try:
        sample = _sample_txt(txt_filepath)
        if not sample:
            print("[TXT Extract] Could not read file.")
            return []

        title_hint = f" (titled '{book_title}')" if book_title else ""
        client = get_llm_client()
        prompt = (
            f"The following are excerpts from a novel{title_hint}.\n"
            f"Identify all MAJOR characters (protagonists and important recurring figures).\n"
            f"Return ONLY a comma-separated list of their names — no explanations, no numbering.\n\n"
            f"Novel excerpts:\n{sample}"
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        text_response = response.choices[0].message.content
        characters = [n.strip() for n in text_response.split(",") if n.strip()]
        print(f"[TXT Extract] Found {len(characters)} characters: {characters}")
        return characters

    except Exception as e:
        print(f"[TXT Extract] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# 1c. Unified entry point — respects CHARACTER_SOURCE setting
# ---------------------------------------------------------------------------

def get_major_character_names(
    book_title: str,
    txt_filepath: str = "",
    source: str = None,
) -> list[str]:
    """
    Unified character-name resolver.

    Args:
        book_title:   Title of the book (used for Wiki and as a hint for LLM).
        txt_filepath: Path to the local .txt file (required for txt_extract / auto).
        source:       Override CHARACTER_SOURCE env var for this call.
                      One of: "wiki" | "txt_extract" | "auto"

    Returns:
        List of character name strings.
    """
    strategy = (source or CHARACTER_SOURCE).lower()

    if strategy == "wiki":
        return get_major_character_names_from_wiki(book_title)

    if strategy == "txt_extract":
        if not txt_filepath:
            print("[CharacterSource] txt_extract requires txt_filepath. Falling back to wiki.")
            return get_major_character_names_from_wiki(book_title)
        return get_major_character_names_from_txt(txt_filepath, book_title)

    # strategy == "auto" (default)
    wiki_results = get_major_character_names_from_wiki(book_title)
    if wiki_results:
        print(f"[CharacterSource] auto → wiki succeeded ({len(wiki_results)} chars).")
        return wiki_results
    print("[CharacterSource] auto → wiki returned 0 results, falling back to txt_extract.")
    if txt_filepath:
        return get_major_character_names_from_txt(txt_filepath, book_title)
    return []


# ---------------------------------------------------------------------------
# 2. Vector index
# ---------------------------------------------------------------------------

def build_book_index(txt_filepath: str, progress_cb=None, persist_directory: str = None):
    """
    Build (or load) a Chroma vector index for the book.
    progress_cb(stage, n, total, message) is called at each step.
    """
    if persist_directory is None:
        book_name = os.path.splitext(os.path.basename(txt_filepath))[0]
        persist_directory = os.path.join(
            os.path.dirname(os.path.abspath(txt_filepath)),
            f"chroma_db_{book_name}",
        )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
        if progress_cb:
            progress_cb("loading_existing", 1, 1, "Loading existing vector index…")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    if progress_cb:
        progress_cb("loading_text", 0, 1, "Loading book text…")

    loader = TextLoader(txt_filepath, encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    total = len(splits)

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    batch_size = 50
    for i in range(0, total, batch_size):
        batch = splits[i : i + batch_size]
        vectorstore.add_documents(batch)
        done = min(i + batch_size, total)
        if progress_cb:
            progress_cb("embedding", done, total, f"Embedding chunks {done} / {total}…")

    return vectorstore


# ---------------------------------------------------------------------------
# 3. RAG retrieval
# ---------------------------------------------------------------------------

def get_character_situations(vectorstore, character_name: str, k: int = 6) -> list[str]:
    """Retrieve the top-k chunks most relevant to scenes featuring `character_name`."""
    query = (
        f"Describe a specific scene involving {character_name}, "
        f"including their location, actions, and what they are wearing."
    )
    relevant_docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in relevant_docs]


def get_scenario_summaries(vectorstore, character_name: str) -> list[dict]:
    """
    Retrieve k=6 RAG scenes for the character and ask the LLM to distill each
    into a short one-sentence label.
    Returns: [{"label": str, "context": str}, ...]
    """
    situations = get_character_situations(vectorstore, character_name, k=6)
    client = get_llm_client()
    summaries = []
    for ctx in situations:
        prompt = (
            f"In one short sentence (max 12 words), describe what is happening in this scene "
            f"involving {character_name}. Return ONLY the sentence, nothing else.\n\n"
            f"Scene:\n{ctx[:2000]}"
        )
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        label = resp.choices[0].message.content.strip().strip('"').strip("'")
        summaries.append({"label": label, "context": ctx})
    return summaries


# ---------------------------------------------------------------------------
# 4. Character analysis
# ---------------------------------------------------------------------------

def analyze_character(book_text: str, character_name: str) -> str:
    """Extract a structured character description from a book excerpt."""
    client = get_llm_client()
    prompt = f"""Analyze the character '{character_name}' from the book. Extract and describe:
- Physical appearance (hair colour, eye colour, height, build, approximate age)
- Clothing and accessories typically worn
- Typical locations where they appear
- Personality traits and mannerisms

Book excerpt:
{book_text[:5000]}"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 5. Actor casting
# ---------------------------------------------------------------------------

def cast_character_with_actor(
    character_name: str, 
    character_description: str, 
    industry: str = "hollywood",
    genre: str = "",
    decade: str = "2026"
) -> str:
    """Suggest a real-world actor from the given industry and decade to portray the character in a specific genre."""
    client = get_llm_client()
    genre_context = f"This is for a {genre} adaptation." if genre else ""
    decade_context = f"The production is set in/filmed during the {decade}s." if decade and decade != "2026" else "The production is modern (2026)."

    prompt = f"""Given the character '{character_name}' with the following description:
{character_description}

{genre_context}
{decade_context}

Cast an age-appropriate real-world actor from the {industry} industry to play this role.
If the decade is in the past, pick an actor who was active and the correct age DURING that decade.
If the decade is modern, pick a currently active actor.
Return ONLY the name of the actor, nothing else."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 6. Prompt generation
# ---------------------------------------------------------------------------

def generate_prompt_for_scenario(
    character_name: str,
    description: str,
    scenario_context: str,
    actor_name: str = "",
    genre: str = "",
    decade: str = "2026",
    gender: str = "",
    race: str = "",
    age: str = "",
) -> str:
    """
    Build a Z-Image-Turbo / Stable Diffusion prompt for one character scene.
    Uses the (possibly user-edited) description and actor name.
    """
    client = get_llm_client()

    # Create override block
    overrides = []
    if gender: overrides.append(f"Gender: {gender}")
    if race: overrides.append(f"Race/Ethnicity: {race}")
    if age: overrides.append(f"Age: {age}")
    override_text = "\n".join(overrides)

    # Extract scene-specific details with genre adaptation for visuals
    genre_context = f" (Adapted specifically for the {genre} genre)" if genre else ""
    extract_prompt = (
        f"Given this book scene{genre_context}, identify the location and what {character_name} is doing.\n"
        f"Then, describe {character_name}'s CLOTHING and HAIRSTYLE as they would appear in a {genre if genre else 'realistic'} adaptation.\n"
        f"IMPORTANT: The clothing and hair must reflect the {genre} genre, but the facial features, age, and ethnicity must remain consistent with a realistic human portrayal of the character.\n\n"
        f"Scene: {scenario_context}"
    )
    scene_resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": extract_prompt}]
    )
    scene_details = scene_resp.choices[0].message.content

    actor_instruction = (
        f"The character's face should closely resemble the actor: {actor_name}."
        if actor_name
        else ""
    )

    genre_instruction = f"Genre: {genre}" if genre else ""
    decade_instruction = f"Visual Style: {decade}s cinematography and fashion" if decade and decade != "2026" else "Visual Style: Modern cinematic hyper-realistic"

    prompt_template = f"""Create a highly detailed image prompt for Z-Image-Turbo.

Character: {character_name}
{genre_instruction}
{decade_instruction}
{actor_instruction}

Core Identity (MUST REMAIN TRUE):
- Physical Overrides: {override_text if override_text else "None"}
- Base Description: {description}

Visual Adaptation (CLOTHING, HAIR, ENVIRONMENT):
- Genre Adaptation: The character's CLOTHING and HAIRSTYLE must be fully modified to fit the '{genre}' genre.
- Environment: The background and visuals must reflect the '{genre}' style.
- Scene Context: {scene_details}

Rules:
- STRICTLY MAINTAIN: expression, age, ethnicity, and facial structure from the core identity.
- FULLY CHANGE: clothing, hair, and lighting/visual effects to match the {genre} genre.
- Style: Photorealistic, cinematic, high detail 8k.
- Atmosphere: Vivid and atmospheric based on the scene context.

Return ONLY the image prompt text, nothing else."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt_template}]
    )
    return response.choices[0].message.content
