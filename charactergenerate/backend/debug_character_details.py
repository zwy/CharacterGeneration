import sys
import json

backend_path = r'k:\UnityProjects\AgenticAiProjects\LLM_annotate\CharacterGenerationFromBook\backend'
if backend_path not in sys.path:
    sys.path.append(backend_path)

from character_gen import (
    build_book_index,
    analyze_character,
    get_character_situations,
    get_scenario_summaries,
)

# ---------------------------------------------------------------------------
# 配置：改成你实际的文件路径和角色名
# ---------------------------------------------------------------------------
TXT_FILE       = r'C:\path\to\your_novel.txt'   # ← 改这里
CHARACTER_NAME = "主角名字"                       # ← 改这里
# ---------------------------------------------------------------------------


def progress_cb(stage, n, total, msg):
    print(f"  [{stage}] {n}/{total} - {msg}")


def test_character_details():
    print("=" * 60)
    print(f"[Step 1] Building / loading vector index for:")
    print(f"         {TXT_FILE}")
    print("=" * 60)

    try:
        vs = build_book_index(TXT_FILE, progress_cb=progress_cb)
        print("[OK] Vector index ready.\n")
    except Exception:
        import traceback
        print("[ERROR] Failed to build index:")
        traceback.print_exc()
        return

    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"[Step 2] analyze_character() — '{CHARACTER_NAME}'")
    print("=" * 60)
    print("Analyzing character... this may take a moment.")

    try:
        situations = get_character_situations(vs, CHARACTER_NAME, k=1)
        context_text = situations[0] if situations else ""
        if not context_text:
            print("[WARN] No relevant scenes found in the index for this character.")
        description = analyze_character(context_text, CHARACTER_NAME)
        print("\n--- Character Description ---")
        print(description)
        print()
    except Exception:
        import traceback
        print("[ERROR] analyze_character failed:")
        traceback.print_exc()
        description = ""

    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"[Step 3] get_scenario_summaries() — '{CHARACTER_NAME}'")
    print("=" * 60)
    print("Retrieving and summarising book scenes... this may take a moment.")

    try:
        scenarios = get_scenario_summaries(vs, CHARACTER_NAME)
        print(f"\nFound {len(scenarios)} scenario(s):")
        for i, s in enumerate(scenarios, 1):
            print(f"\n  [{i}] {s['label']}")
            print(f"      (context preview: {s['context'][:120].strip()}...)")
    except Exception:
        import traceback
        print("[ERROR] get_scenario_summaries failed:")
        traceback.print_exc()


if __name__ == "__main__":
    test_character_details()
