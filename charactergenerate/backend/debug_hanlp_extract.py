"""
debug_hanlp_extract.py
调试 HanLP NER + LLM 精炼策略的独立测试脚本。

用法：
  python debug_hanlp_extract.py \\
      --txt  "C:\\path\\to\\novel.txt" \\
      --book "小说名称" \\
      --base-url http://localhost:11434/v1 \\
      --model Gemma4E4B:latest
"""

import sys
import os
import time
import argparse
import pprint

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

parser = argparse.ArgumentParser(description="调试 HanLP NER + LLM 精炼策略")
parser.add_argument("--txt",      default=None)
parser.add_argument("--book",     default=None)
parser.add_argument("--base-url", default=None)
parser.add_argument("--model",    default=None)
parser.add_argument("--max-candidates", type=int, default=80)
args, _ = parser.parse_known_args()

if args.base_url:
    os.environ["LLM_BASE_URL"] = args.base_url
if args.model:
    os.environ["LLM_MODEL"] = args.model

import character_gen as cg
from character_gen import (
    _read_full_txt,
    _extract_names_hanlp,
    _parse_hanlp_ner_result,
    get_major_character_names_from_txt_hanlp,
    get_major_character_names_from_txt,
    get_major_character_names,
    LLM_BASE_URL,
    LLM_MODEL,
)

TXT_FILE       = args.txt  or r'C:\path\to\your_novel.txt'
BOOK_NAME      = args.book or "你的小说名"
MAX_CANDIDATES = args.max_candidates


def _check_file() -> bool:
    if not os.path.exists(TXT_FILE):
        print(f"[ERROR] File not found: {TXT_FILE}")
        return False
    return True


def print_config():
    print(f"  txt 文件  : {TXT_FILE}")
    print(f"  书名      : {BOOK_NAME or '(未指定)'}")
    print(f"  LLM URL   : {LLM_BASE_URL}")
    print(f"  LLM 模型  : {LLM_MODEL}")
    print(f"  候选上限  : {MAX_CANDIDATES}")
    print()


def test_hanlp_output_format():
    """
    Step 0: 用前 10 句话测试 HanLP 模型的实际返回格式。
    这一步不依赖任何解析逻辑，直接打印原始 ner() 输出，
    帮助诊断 tensor / 格式问题。
    """
    print("=" * 60)
    print("[Step 0] HanLP 原始输出格式诊断")
    print("=" * 60)

    try:
        import hanlp  # type: ignore
        import re

        sample = "刘小静推开宿舍的门。付筱竹坐在床上看书。秦大爷从门房走出来。"
        sentences = re.split(r"[。！？!?\n]+", sample)
        sentences = [s.strip() for s in sentences if s.strip()]
        print(f"测试句子（{len(sentences)} 条）: {sentences}")
        print()

        _MODEL_CANDIDATES = [
            "MSRA_NER_ELECTRA_SMALL_ZH",
            "CTB9_NER_ELECTRA_SMALL",
        ]

        for model_attr in _MODEL_CANDIDATES:
            try:
                model_id = getattr(hanlp.pretrained.ner, model_attr, None)
                if model_id is None:
                    print(f"  {model_attr}: 不在 hanlp.pretrained.ner 中，跳过")
                    continue
                print(f"  加载模型: {model_attr} ...")
                ner = hanlp.load(model_id)

                print("  --- 输入: List[str] (batch) ---")
                result_batch = ner(sentences)
                print(f"  返回类型 : {type(result_batch)}")
                print(f"  返回内容 :")
                pprint.pprint(result_batch, indent=4)
                print()

                print("  --- 输入: str (单句) ---")
                result_single = ner(sentences[0])
                print(f"  返回类型 : {type(result_single)}")
                print(f"  返回内容 :")
                pprint.pprint(result_single, indent=4)
                print()

                break  # 找到第一个可用模型就够了
            except Exception as e:
                print(f"  {model_attr} 加载/运行失败: {e}")
                print()

    except ImportError:
        print("  hanlp 未安装，Step 0 跳过。")
    except Exception as e:
        import traceback
        traceback.print_exc()

    print("[OK] Step 0 诊断完成，请把上方输出贴给开发者。")
    print()


def test_read_full() -> bool:
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


def test_ner_extraction() -> bool:
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
        print("[WARN] NER 未提取到任何候选名。")
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
    print()
    print("=" * 60)
    print("[Step 3] HanLP + LLM 精炼")
    print("=" * 60)

    if not _check_file():
        return []

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
        if chars:
            print("--- 角色列表 ---")
            for i, c in enumerate(chars, 1):
                print(f"  {i:>2}. {c}")
            print()
            print("[OK] HanLP 策略测试通过。")
        else:
            print("[WARN] 未提取到角色，请检查 LLM 配置。")
            print(f"       当前 LLM_BASE_URL = {LLM_BASE_URL}")
            print(f"       当前 LLM_MODEL    = {LLM_MODEL}")
        return chars
    except Exception:
        import traceback
        traceback.print_exc()
        return []


def test_compare_strategies():
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
    print(f"仅 hanlp 有 ({len(only_hanlp)} 个) : {', '.join(only_hanlp) or '(无)'}")
    print(f"仅 txt_extract 有 ({len(only_txt)} 个) : {', '.join(only_txt) or '(无)'}")


def test_auto_mode():
    print()
    print("=" * 60)
    print("[Step 5] auto 模式回落测试")
    print("=" * 60)

    if not _check_file():
        return

    fake_title = BOOK_NAME or "LocalNovelNotOnWikipedia_XYZ"
    print(f"书名 : '{fake_title}'")
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
    print_config()

    # Step 0: 诊断 HanLP 实际输出格式（不依赖解析逻辑）
    test_hanlp_output_format()

    ok = test_read_full()
    if not ok:
        sys.exit(1)

    ner_ok = test_ner_extraction()
    if not ner_ok:
        print("\n[WARN] NER 步骤失败，跳过后续测试。")
        sys.exit(1)

    test_hanlp_strategy()
    test_compare_strategies()
    test_auto_mode()
