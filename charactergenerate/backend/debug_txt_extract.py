import sys
import os
import json

backend_path = r'k:\UnityProjects\AgenticAiProjects\LLM_annotate\CharacterGenerationFromBook\backend'
if backend_path not in sys.path:
    sys.path.append(backend_path)

from character_gen import (
    _sample_txt,
    get_major_character_names_from_txt,
    get_major_character_names,
)

# ---------------------------------------------------------------------------
# 配置：改成你本地 txt 文件的实际路径和书名
# ---------------------------------------------------------------------------
TXT_FILE  = r'C:\path\to\your_novel.txt'   # ← 改这里
BOOK_NAME = "你的小说名"                        # ← 或留空白（仅作提示）
# ---------------------------------------------------------------------------


def test_sample():
    """Step 1: 验证采样逻辑，打印采样片段的字符数和内容预览"""
    print("=" * 60)
    print("[Step 1] Testing _sample_txt()")
    print("=" * 60)

    if not os.path.exists(TXT_FILE):
        print(f"[ERROR] File not found: {TXT_FILE}")
        return False

    sample = _sample_txt(TXT_FILE, total_chars=15000)
    if not sample:
        print("[ERROR] _sample_txt returned empty string.")
        return False

    print(f"Sample length : {len(sample)} chars")
    print(f"File size     : {os.path.getsize(TXT_FILE):,} bytes")
    print()
    print("--- Sample preview (first 300 chars) ---")
    print(sample[:300])
    print("...")
    print("[OK] Sampling looks good.")
    return True


def test_txt_extract():
    """Step 2: 直接调用 txt_extract 策略提取角色"""
    print()
    print("=" * 60)
    print("[Step 2] Testing get_major_character_names_from_txt()")
    print("=" * 60)

    if not os.path.exists(TXT_FILE):
        print(f"[ERROR] File not found: {TXT_FILE}")
        return

    try:
        print(f"Extracting characters from '{TXT_FILE}' (book: '{BOOK_NAME}')...")
        chars = get_major_character_names_from_txt(TXT_FILE, BOOK_NAME)
        print(f"Found {len(chars)} character(s):")
        for i, c in enumerate(chars, 1):
            print(f"  {i:>2}. {c}")
    except Exception:
        import traceback
        traceback.print_exc()


def test_auto_fallback():
    """Step 3: 测试 auto 模式——显然不在 Wikipedia 的书名应自动回落到 txt_extract"""
    print()
    print("=" * 60)
    print("[Step 3] Testing get_major_character_names() in auto mode")
    print("=" * 60)

    fake_title = BOOK_NAME or "LocalNovelNotOnWikipedia_XYZ"
    try:
        print(f"Calling get_major_character_names(book_title='{fake_title}', source='auto')...")
        chars = get_major_character_names(
            book_title=fake_title,
            txt_filepath=TXT_FILE,
            source="auto",
        )
        print(f"Found {len(chars)} character(s):")
        for i, c in enumerate(chars, 1):
            print(f"  {i:>2}. {c}")
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ok = test_sample()
    if ok:
        test_txt_extract()
    test_auto_fallback()
