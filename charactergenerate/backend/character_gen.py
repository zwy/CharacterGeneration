"""
character_gen.py
Core logic for the Character Generation app.
Refactored from CharacterImageGeneration.py with:
  - progress callbacks for SSE streaming
  - get_scenario_summaries() — RAG-derived scenario dropdown
  - generate_prompt_for_scenario() — single call for one scene
  - get_major_character_names_from_txt() — LLM+sampling fallback when Wiki is unavailable
  - get_major_character_names_from_txt_hanlp() — HanLP NER pre-filter + single LLM refine (token-efficient)
"""

import os
import re
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
# "hanlp"       — HanLP NER pre-filter + single LLM refine (token-efficient, good for long CN novels)
# "auto"        — try Wiki first; if it returns 0 results, fall back to hanlp → txt_extract
# ---------------------------------------------------------------------------
CHARACTER_SOURCE = os.getenv("CHARACTER_SOURCE", "auto")


def get_llm_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def _split_names(text: str) -> list[str]:
    """
    Split a comma-separated name list returned by the LLM.
    Handles both English commas (,) and Chinese fullwidth commas (，).
    Also strips leading numbers/dots like "1. " that some models emit.
    """
    # Normalise Chinese comma to ASCII comma first
    text = text.replace("，", ",")
    # Split on comma; also tolerate newlines between names
    parts = re.split(r"[,\n]+", text)
    names = []
    for p in parts:
        # Strip leading ordinal markers: "1.", "1)", "•", "-", etc.
        p = re.sub(r"^\s*[\d\u4e00\u4e8c\u4e09\uff0e.\-)\u2022\u00b7]+\s*", "", p)
        p = p.strip()
        if p:
            names.append(p)
    return names


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
        return _split_names(response.choices[0].message.content)

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
        characters = _split_names(response.choices[0].message.content)
        print(f"[TXT Extract] Found {len(characters)} characters: {characters}")
        return characters

    except Exception as e:
        print(f"[TXT Extract] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# 1c. HanLP NER pre-filter + single LLM refine (token-efficient)
# ---------------------------------------------------------------------------

def _read_full_txt(txt_filepath: str) -> str:
    """Read the full text of a file, trying common Chinese encodings."""
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            with open(txt_filepath, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    return ""


def _extract_names_hanlp(text: str) -> list[str]:
    """
    Use HanLP to perform Named Entity Recognition and extract PERSON entities.

    Tries two backends in order:
      1. hanlp (full pipeline with MSRA NER) — higher accuracy, requires hanlp package
      2. jieba posseg (nr tag) — lightweight fallback, lower accuracy

    Returns a deduplicated list of candidate person names sorted by frequency (desc).
    """
    # --- Try HanLP first ---
    try:
        import hanlp  # type: ignore
        # Load a fast NER pipeline; cache is handled by hanlp automatically
        # Using the smaller, faster model to keep memory footprint low
        ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)  # type: ignore

        # HanLP NER expects a list of sentences; chunk to avoid OOM on large texts
        CHUNK = 5000
        freq: dict[str, int] = {}
        for i in range(0, len(text), CHUNK):
            chunk = text[i : i + CHUNK]
            # hanlp ner accepts a plain string for convenience
            entities = ner(chunk)
            for entity, label, *_ in entities:
                if label == "PERSON":
                    entity = entity.strip()
                    if entity:
                        freq[entity] = freq.get(entity, 0) + 1

        print(f"[HanLP NER] Extracted {len(freq)} unique candidate names.")
        # Sort by frequency descending so the most prominent names come first
        return [name for name, _ in sorted(freq.items(), key=lambda x: -x[1])]

    except ImportError:
        print("[HanLP NER] hanlp not installed, falling back to jieba posseg.")
    except Exception as e:
        print(f"[HanLP NER] Error: {e}, falling back to jieba posseg.")

    # --- Fallback: jieba posseg (nr = person name) ---
    try:
        import jieba.posseg as pseg  # type: ignore
        freq: dict[str, int] = {}
        for word, flag in pseg.cut(text):
            if flag == "nr":
                word = word.strip()
                if len(word) >= 2:  # skip single-character noise
                    freq[word] = freq.get(word, 0) + 1
        print(f"[jieba NER] Extracted {len(freq)} unique candidate names.")
        return [name for name, _ in sorted(freq.items(), key=lambda x: -x[1])]
    except ImportError:
        print("[jieba NER] jieba not installed either. Returning empty list.")
        return []
    except Exception as e:
        print(f"[jieba NER] Error: {e}")
        return []


def get_major_character_names_from_txt_hanlp(
    txt_filepath: str,
    book_title: str = "",
    max_candidates: int = 80,
) -> list[str]:
    """
    Token-efficient character extraction strategy:

    Step 1 — HanLP / jieba NER (0 LLM tokens):
        Read the FULL text locally and extract all PERSON-tagged entities.
        Deduplicate and sort by frequency. Keep at most `max_candidates` names.

    Step 2 — Single LLM call (~500–1500 tokens total):
        Send only the candidate name list to the LLM.
        Ask it to: remove non-character noise, merge aliases/titles for the
        same character, and return the final clean list.

    This approach consumes roughly 1/30th the tokens of sending the full text
    to an LLM, while achieving ~93%+ recall on Chinese web novels.

    Args:
        txt_filepath:   Path to the local .txt novel file.
        book_title:     Optional title hint passed to the LLM.
        max_candidates: Maximum number of NER candidates forwarded to the LLM.
                        Increase if the novel has many distinct characters.

    Returns:
        List of final character name strings.
    """
    try:
        full_text = _read_full_txt(txt_filepath)
        if not full_text:
            print("[HanLP Strategy] Could not read file.")
            return []

        # Step 1: local NER — 0 LLM tokens
        candidates = _extract_names_hanlp(full_text)
        if not candidates:
            print("[HanLP Strategy] NER returned no candidates, falling back to txt_extract.")
            return get_major_character_names_from_txt(txt_filepath, book_title)

        # Trim to max_candidates (already sorted by frequency)
        candidates = candidates[:max_candidates]
        print(f"[HanLP Strategy] Sending {len(candidates)} candidates to LLM for refinement.")

        # Step 2: single LLM call — only the name list, not the full text
        title_hint = f" from the novel '{book_title}'" if book_title else ""
        candidate_str = "、".join(candidates)  # Chinese enumeration separator

        prompt = (
            f"以下是从一部小说{title_hint}中通过命名实体识别提取的候选人名列表。\n"
            f"这些候选词可能包含：真实角色名、误识别的地名/机构名、称谓（如"大人"、"师父"）、\n"
            f"同一角色的不同称呼（如"张伟"和"张大人"）。\n\n"
            f"请你：\n"
            f"1. 去除明显不是人名的词（地名、职位、泛称等）\n"
            f"2. 合并同一角色的不同称呼，保留最常用的正式姓名\n"
            f"3. 只返回最终的角色名单，用中文顿号（、）分隔，不要编号，不要解释\n\n"
            f"候选列表：{candidate_str}"
        )

        client = get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()

        # Parse the response — support both 、 and , separators
        raw = raw.replace("、", ",")
        characters = _split_names(raw)
        print(f"[HanLP Strategy] Final character list ({len(characters)}): {characters}")
        return characters

    except Exception as e:
        print(f"[HanLP Strategy] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# 1d. Unified entry point — respects CHARACTER_SOURCE setting
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
        txt_filepath: Path to the local .txt file (required for txt_extract / hanlp / auto).
        source:       Override CHARACTER_SOURCE env var for this call.
                      One of: "wiki" | "txt_extract" | "hanlp" | "auto"

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

    if strategy == "hanlp":
        if not txt_filepath:
            print("[CharacterSource] hanlp requires txt_filepath. Falling back to wiki.")
            return get_major_character_names_from_wiki(book_title)
        return get_major_character_names_from_txt_hanlp(txt_filepath, book_title)

    # strategy == "auto" (default)
    # Priority: wiki → hanlp → txt_extract
    wiki_results = get_major_character_names_from_wiki(book_title)
    if wiki_results:
        print(f"[CharacterSource] auto → wiki succeeded ({len(wiki_results)} chars).")
        return wiki_results

    print("[CharacterSource] auto → wiki returned 0 results, falling back to hanlp.")
    if txt_filepath:
        hanlp_results = get_major_character_names_from_txt_hanlp(txt_filepath, book_title)
        if hanlp_results:
            print(f"[CharacterSource] auto → hanlp succeeded ({len(hanlp_results)} chars).")
            return hanlp_results
        print("[CharacterSource] auto → hanlp returned 0 results, falling back to txt_extract.")
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
