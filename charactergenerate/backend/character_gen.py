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
# ---------------------------------------------------------------------------
CHARACTER_SOURCE = os.getenv("CHARACTER_SOURCE", "auto")


def get_llm_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def _split_names(text: str) -> list[str]:
    text = text.replace("，", ",")
    parts = re.split(r"[,\n]+", text)
    names = []
    for p in parts:
        p = re.sub(r"^\s*[\d\u4e00\u4e8c\u4e09\uff0e.\-)\u2022\u00b7]+\s*", "", p)
        p = p.strip()
        if p:
            names.append(p)
    return names


# ---------------------------------------------------------------------------
# 1a. Wikipedia
# ---------------------------------------------------------------------------

def get_major_character_names_from_wiki(book_title: str) -> list[str]:
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
# 1b. TXT sampling
# ---------------------------------------------------------------------------

def _sample_txt(txt_filepath: str, total_chars: int = 15000) -> str:
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
    head = full_text[:chunk]
    mid_start = (length // 2) - (chunk // 2)
    middle = full_text[mid_start: mid_start + chunk]
    tail = full_text[length - chunk:]

    return (
        "=== Beginning ===\n" + head +
        "\n\n=== Middle ===\n" + middle +
        "\n\n=== End ===\n" + tail
    )


def get_major_character_names_from_txt(txt_filepath: str, book_title: str = "") -> list[str]:
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
# 1c. HanLP NER pre-filter + single LLM refine
# ---------------------------------------------------------------------------

def _read_full_txt(txt_filepath: str) -> str:
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            with open(txt_filepath, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    return ""


def _parse_hanlp_ner_result(result, freq: dict) -> None:
    """
    HanLP 2.1.x NER models have inconsistent return formats depending on
    whether input was a single string or a list, and which model is used.
    This function handles all known formats robustly:

    Format A — List[List[Tuple[str, str, int, int]]]  (batch input, flat spans per sentence)
      [[('张伟', 'PERSON', 0, 2), ('北京', 'LOCATION', 3, 5)], [...]]

    Format B — List[Tuple[str, str, int, int]]  (single sentence input)
      [('张伟', 'PERSON', 0, 2), ('北京', 'LOCATION', 3, 5)]

    Format C — List[str]  (just entity text, no label — some lite models)
      ['张伟', '李明']

    Format D — dict with 'ner' key  (some pipeline outputs)
      {'ner': [[('张伟', 'PERSON', 0, 2)]]}
    """
    if result is None:
        return

    # Format D: dict output from pipeline
    if isinstance(result, dict):
        result = result.get("ner", result.get("NER", []))

    if not result:
        return

    first = result[0]

    # Format A: outer list contains inner lists (batch of sentences)
    if isinstance(first, list):
        for sent_spans in result:
            for span in sent_spans:
                _parse_single_span(span, freq)
        return

    # Format B: outer list contains tuples/lists directly (single sentence)
    if isinstance(first, (tuple, list)) and len(first) >= 2 and not isinstance(first[0], (tuple, list)):
        for span in result:
            _parse_single_span(span, freq)
        return

    # Format C: outer list contains plain strings
    if isinstance(first, str):
        for entity in result:
            entity = str(entity).strip()
            if entity:
                freq[entity] = freq.get(entity, 0) + 1
        return

    # Unknown format: attempt recursive descent
    try:
        for item in result:
            _parse_hanlp_ner_result(item, freq)
    except Exception:
        pass


def _parse_single_span(span, freq: dict) -> None:
    """Parse a single NER span regardless of whether it's a tuple or list."""
    try:
        if isinstance(span, (tuple, list)) and len(span) >= 2:
            entity = str(span[0]).strip()
            label = str(span[1])
            if label == "PERSON" and entity:
                freq[entity] = freq.get(entity, 0) + 1
    except Exception:
        pass


def _extract_names_hanlp(text: str) -> list[str]:
    """
    Use HanLP NER to extract PERSON entities from text.
    Falls back to jieba posseg if HanLP is unavailable or fails.
    """
    try:
        import hanlp  # type: ignore

        # Split text into sentences — HanLP NER works on sentence lists
        _SENT_RE = re.compile(r"[。！？!?\n]+")
        sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
        if not sentences:
            sentences = [text]

        _MODEL_CANDIDATES = [
            "MSRA_NER_ELECTRA_SMALL_ZH",
            "CTB9_NER_ELECTRA_SMALL",
        ]

        ner = None
        loaded_model_name = None
        for model_attr in _MODEL_CANDIDATES:
            try:
                model_id = getattr(hanlp.pretrained.ner, model_attr, None)
                if model_id is None:
                    print(f"[HanLP NER] {model_attr} not in hanlp.pretrained.ner, skipping.")
                    continue
                ner = hanlp.load(model_id)
                loaded_model_name = model_attr
                print(f"[HanLP NER] Loaded model: {model_attr}")
                break
            except Exception as model_err:
                print(f"[HanLP NER] Skipping {model_attr}: {model_err}")

        if ner is None:
            raise RuntimeError("All HanLP NER models failed to load.")

        BATCH = 32  # smaller batch = less likely to OOM
        freq: dict[str, int] = {}
        for i in range(0, len(sentences), BATCH):
            batch = sentences[i: i + BATCH]
            try:
                result = ner(batch)
                _parse_hanlp_ner_result(result, freq)
            except Exception as batch_err:
                # Try sentence-by-sentence as last resort
                for sent in batch:
                    try:
                        result = ner(sent)  # single string fallback
                        _parse_hanlp_ner_result(result, freq)
                    except Exception:
                        pass

        print(f"[HanLP NER] Extracted {len(freq)} unique names (model: {loaded_model_name}).")
        return [name for name, _ in sorted(freq.items(), key=lambda x: -x[1])]

    except ImportError:
        print("[HanLP NER] hanlp not installed, falling back to jieba posseg.")
    except Exception as e:
        print(f"[HanLP NER] Error: {e}, falling back to jieba posseg.")

    # --- Fallback: jieba posseg ---
    try:
        import jieba.posseg as pseg  # type: ignore
        freq: dict[str, int] = {}
        for word, flag in pseg.cut(text):
            if flag == "nr":
                word = word.strip()
                if len(word) >= 2:
                    freq[word] = freq.get(word, 0) + 1
        print(f"[jieba NER] Extracted {len(freq)} unique candidate names.")
        return [name for name, _ in sorted(freq.items(), key=lambda x: -x[1])]
    except ImportError:
        print("[jieba NER] jieba not installed either.")
        return []
    except Exception as e:
        print(f"[jieba NER] Error: {e}")
        return []


def get_major_character_names_from_txt_hanlp(
    txt_filepath: str,
    book_title: str = "",
    max_candidates: int = 80,
) -> list[str]:
    try:
        full_text = _read_full_txt(txt_filepath)
        if not full_text:
            print("[HanLP Strategy] Could not read file.")
            return []

        candidates = _extract_names_hanlp(full_text)
        if not candidates:
            print("[HanLP Strategy] NER returned no candidates, falling back to txt_extract.")
            return get_major_character_names_from_txt(txt_filepath, book_title)

        candidates = candidates[:max_candidates]
        print(f"[HanLP Strategy] Sending {len(candidates)} candidates to LLM for refinement.")

        title_hint = f" from the novel '{book_title}'" if book_title else ""
        candidate_str = "、".join(candidates)

        prompt = (
            f"以下是从一部小说{title_hint}中通过命名实体识别提取的候选人名列表。\n"
            f"这些候选词可能包含：真实角色名、误识别的地名/机构名、称谓（如\"大人\"、\"师父\"）、\n"
            f"同一角色的不同称呼（如\"张伟\"和\"张大人\"）。\n\n"
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
        raw = raw.replace("、", ",")
        characters = _split_names(raw)
        print(f"[HanLP Strategy] Final character list ({len(characters)}): {characters}")
        return characters

    except Exception as e:
        print(f"[HanLP Strategy] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# 1d. Unified entry point
# ---------------------------------------------------------------------------

def get_major_character_names(
    book_title: str,
    txt_filepath: str = "",
    source: str = None,
) -> list[str]:
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

    # auto: wiki → hanlp → txt_extract
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
        batch = splits[i: i + batch_size]
        vectorstore.add_documents(batch)
        done = min(i + batch_size, total)
        if progress_cb:
            progress_cb("embedding", done, total, f"Embedding chunks {done} / {total}…")

    return vectorstore


# ---------------------------------------------------------------------------
# 3. RAG retrieval
# ---------------------------------------------------------------------------

def get_character_situations(vectorstore, character_name: str, k: int = 6) -> list[str]:
    query = (
        f"Describe a specific scene involving {character_name}, "
        f"including their location, actions, and what they are wearing."
    )
    relevant_docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in relevant_docs]


def get_scenario_summaries(vectorstore, character_name: str) -> list[dict]:
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
    client = get_llm_client()

    overrides = []
    if gender: overrides.append(f"Gender: {gender}")
    if race: overrides.append(f"Race/Ethnicity: {race}")
    if age: overrides.append(f"Age: {age}")
    override_text = "\n".join(overrides)

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
        if actor_name else ""
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
