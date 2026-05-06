import sys
import os
import time

backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from character_gen import (
    _read_full_txt,
    _extract_names_hanlp,
    get_major_character_names_from_txt_hanlp,
    get_major_character_names_from_txt,
    get_major_character_names,
)

# ---------------------------------------------------------------------------
# 配置：改成你本地 txt 文件的实际路径和书名
# ---------------------------------------------------------------------------
# TXT_FILE = r'c:\Users\Administrator\Desktop\淫荡少妇白洁.txt'   # ← 改这里
# BOOK_NAME = "淫荡少妇白洁"                        # ← 或留空白（仅作提示）
TXT_FILE = r'c:\Users\Administrator\Desktop\女生宿舍门房秦大爷的故事.txt'   # ← 改这里
BOOK_NAME = "女生宿舍门房秦大爷的故事"                        # ← 或留空白（仅作提示）
MAX_CANDIDATES = 80                        # NER 候选名最大保留数量
# ---------------------------------------------------------------------------


def _check_file() -> bool:
    if not os.path.exists(TXT_FILE):
        print(f"[ERROR] File not found: {TXT_FILE}")
        print("请修改脚本顶部的 TXT_FILE 路径。")
        return False
    return True


def test_read_full():
    """验证能否完整读取文件并打印基本信息"""
    print("=" * 60)
    print("[Step 1] 验证文件读取 _read_full_txt()")
    print("=" * 60)

    if not _check_file():
        return False

    t0 = time.time()
    full_text = _read_full_txt(TXT_FILE)
    elapsed = time.time() - t0

    if not full_text:
        print("[ERROR] _read_full_txt 返回空内容，请检查文件编码。")
        return False

    size_mb = os.path.getsize(TXT_FILE) / 1024 / 1024
    print(f"文件大小 : {size_mb:.2f} MB")
    print(f"总字数   : {len(full_text):,} 字")
    print(f"读取耗时 : {elapsed:.2f}s")
    print()
    print("--- 文件内容预览（前 200 字）---")
    print(full_text[:200])
    print("...")
    print("[OK] 文件读取成功。")
    return True


def test_ner_extraction():
    """验证 HanLP/jieba NER 提取，0 LLM token"""
    print()
    print("=" * 60)
    print("[Step 2] HanLP / jieba NER 候选名提取 _extract_names_hanlp()")
    print("=" * 60)

    if not _check_file():
        return False

    full_text = _read_full_txt(TXT_FILE)
    if not full_text:
        return False

    print(f"对 {len(full_text):,} 字进行 NER，请稍候...")
    t0 = time.time()
    candidates = _extract_names_hanlp(full_text)
    elapsed = time.time() - t0

    if not candidates:
        print("[WARN] NER 未提取到任何候选名，请检查 hanlp / jieba 是否安装。")
        return False

    print(f"NER 耗时     : {elapsed:.1f}s")
    print(f"候选名数量 : {len(candidates)}")
    print()

    top_n = candidates[:MAX_CANDIDATES]
    print(f"--- Top {len(top_n)} 候选名（按出现频率排序）---")
    for i, name in enumerate(top_n, 1):
        print(f"  {i:>3}. {name}")

    print()
    print("[OK] NER 提取完成。")
    return True


def test_hanlp_strategy():
    """验证完整的 HanLP NER + LLM 精炼流程"""
    print()
    print("=" * 60)
    print("[Step 3] HanLP + LLM 精炼 get_major_character_names_from_txt_hanlp()")
    print("=" * 60)

    if not _check_file():
        return

    print(f"处理文件 : {TXT_FILE}")
    print(f"书名提示 : {BOOK_NAME or '(未指定)'} | 候选名上限 : {MAX_CANDIDATES}")
    print()

    t0 = time.time()
    try:
        chars = get_major_character_names_from_txt_hanlp(
            TXT_FILE, BOOK_NAME, max_candidates=MAX_CANDIDATES
        )
        elapsed = time.time() - t0
        print(f"总耗时 : {elapsed:.1f}s")
        print(f"最终角色数 : {len(chars)}")
        print()
        print("--- 角色列表 ---")
        for i, c in enumerate(chars, 1):
            print(f"  {i:>2}. {c}")
        print()
        print("[OK] HanLP 策略测试通过。")
    except Exception:
        import traceback
        traceback.print_exc()


def test_compare_strategies():
    """对比 hanlp 策略 vs txt_extract 策略，输出差异"""
    print()
    print("=" * 60)
    print("[Step 4] 对比 hanlp vs txt_extract 策略结果")
    print("=" * 60)

    if not _check_file():
        return

    print("… 运行 hanlp 策略 ...")
    t0 = time.time()
    hanlp_chars = get_major_character_names_from_txt_hanlp(
        TXT_FILE, BOOK_NAME, max_candidates=MAX_CANDIDATES
    )
    t_hanlp = time.time() - t0

    print("… 运行 txt_extract 策略 ...")
    t0 = time.time()
    txt_chars = get_major_character_names_from_txt(TXT_FILE, BOOK_NAME)
    t_txt = time.time() - t0

    set_hanlp = set(hanlp_chars)
    set_txt   = set(txt_chars)

    only_hanlp = sorted(set_hanlp - set_txt)
    only_txt   = sorted(set_txt   - set_hanlp)
    common     = sorted(set_hanlp & set_txt)

    print()
    print(f"{'':20s}  {'hanlp':>8}  {'txt_extract':>12}")
    print(f"{'总角色数':20s}  {len(hanlp_chars):>8}  {len(txt_chars):>12}")
    print(f"{'耗时(s)':20s}  {t_hanlp:>8.1f}  {t_txt:>12.1f}")
    print()
    print(f"两种策略都有 ({len(common)} 个) : {', '.join(common) or '(无)'}")
    print()
    print(f"仅 hanlp 有 ({len(only_hanlp)} 个) : {', '.join(only_hanlp) or '(无)'}")
    print()
    print(f"仅 txt_extract 有 ({len(only_txt)} 个) : {', '.join(only_txt) or '(无)'}")


def test_auto_mode():
    """验证 auto 模式是否正确回落到 hanlp"""
    print()
    print("=" * 60)
    print("[Step 5] auto 模式回落测试 get_major_character_names()")
    print("=" * 60)

    if not _check_file():
        return

    fake_title = BOOK_NAME or "LocalNovelNotOnWikipedia_XYZ"
    print(f"书名 : '{fake_title}'（预期 wiki 失败并回落到 hanlp）")
    try:
        chars = get_major_character_names(
            book_title=fake_title,
            txt_filepath=TXT_FILE,
            source="auto",
        )
        print(f"找到 {len(chars)} 个角色:")
        for i, c in enumerate(chars, 1):
            print(f"  {i:>2}. {c}")
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n>>> CharacterGeneration — HanLP NER 策略调试脚本 <<<\n")

    ok = test_read_full()
    if not ok:
        sys.exit(1)

    ner_ok = test_ner_extraction()
    if not ner_ok:
        print("\n[WARN] NER 随直失败，跳过后续测试。")
        sys.exit(1)

    test_hanlp_strategy()
    # test_compare_strategies()
    
    # test_auto_mode()
