# src/pipeline.py
from __future__ import annotations
import os
import json
import argparse
import traceback
from typing import List, Dict, Any, Tuple

# 识别器：你项目里已有这个模块
# - HeuristicRecognizer: 规则版
# - LLMRecognizer: DeepSeek 等大模型版（OpenAI 兼容）
from .recognizer import HeuristicRecognizer, LLMRecognizer

# 数据库（可选）
try:
    from .db import get_engine, insert_person_information
except Exception:
    get_engine = None
    insert_person_information = None



def load_tokens_from_file(path: str) -> List[str]:
    """
    读取一个输入文件，返回 tokens（list[str]）
    支持两种格式：
      1) 仅 tokens： [ "患者", "李某", ... ]
      2) gold 格式： { "tokens": [...], "entities": [...] }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        # 纯 tokens 文件
        tokens = data
    elif isinstance(data, dict) and "tokens" in data and isinstance(data["tokens"], list):
        # gold 文件也可读（只取 tokens）
        tokens = data["tokens"]
    else:
        raise ValueError(f"Unsupported file format: {path}")

    # 基本校验
    if not all(isinstance(t, str) for t in tokens):
        raise ValueError(f"Tokens must be list[str]: {path}")
    return tokens


def get_recognizer(backend: str):
    backend = (backend or "heuristic").lower()
    if backend == "llm":
        return LLMRecognizer()
    elif backend == "heuristic":
        return HeuristicRecognizer()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def call_recognizer(recognizer, tokens: List[str]) -> List[Dict[str, Any]]:
    """
    优先以“列表 tokens”方式调用；若识别器不支持，回退为拼接字符串调用。
    （这样即使你的 recognizer 还没改完，也不会崩）
    """
    # 优先：如果有专门的 recognize_tokens 方法
    if hasattr(recognizer, "recognize_tokens"):
        return recognizer.recognize_tokens(tokens)

    # 其次：尝试直接把 list 传进去
    try:
        return recognizer.recognize(tokens)  # 你的识别器若已支持 list 会走这里
    except TypeError:
        # 回退：拼接为字符串（保证兼容旧版）
        text = "".join(tokens)
        return recognizer.recognize(text)


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def process_one_file(recognizer, in_path: str, out_path: str, source_id: str,
                     engine=None, write_db: bool = True) -> Tuple[bool, str]:
    """
    处理单个文件：
      - 读取 tokens
      - 调用识别器
      - 写出 JSON（tokens + entities）
      - 可选入库（text 为拼接版，entities 为 JSON）
    返回 (成功?, 错误消息或输出路径)
    """
    try:
        tokens = load_tokens_from_file(in_path)
        entities = call_recognizer(recognizer, tokens)

        payload = {
            "tokens": tokens,
            "entities": entities
        }

        ensure_parent_dir(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 可选入库
        if write_db and engine is not None and insert_person_information is not None:
            text_value = "".join(tokens)  # 入库文本用拼接版，方便检索
            insert_person_information(
                engine,
                source_id=source_id,
                source_path=in_path,
                text_value=text_value,
                entities_json=entities,
                model=type(recognizer).__name__.replace("Recognizer", "").lower()  # heuristic / llm
            )

        return True, out_path
    except Exception as e:
        return False, f"{in_path} -> {e}"


def iter_json_files(root: str):
    """递归遍历 root 下所有 .json 文件"""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                yield os.path.join(dirpath, fn)


def main():
    parser = argparse.ArgumentParser(description="Batch PII detection pipeline (process all files in a folder)")
    parser.add_argument("--input", required=True, help="输入文件夹（包含 tokens 或 gold）")
    parser.add_argument("--output", required=True, help="输出文件夹（将生成 tokens+entities 的 JSON）")
    parser.add_argument("--backend", choices=["heuristic", "llm"], default="heuristic", help="识别后端")
    parser.add_argument("--no-db", action="store_true", help="不写入数据库（默认写入）")
    args = parser.parse_args()

    recognizer = get_recognizer(args.backend)

    # 初始化数据库（可选）
    engine = None
    if not args.no_db and get_engine is not None:
        try:
            engine = get_engine()
        except Exception:
            engine = None
            print("⚠️ 连接数据库失败，将跳过入库。")

    in_root = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output)
    os.makedirs(out_root, exist_ok=True)

    total = 0
    ok = 0
    fail = 0
    failures = []

    for in_path in iter_json_files(in_root):
        total += 1
        rel_path = os.path.relpath(in_path, in_root)
        out_path = os.path.join(out_root, rel_path)
        # source_id 使用相对路径（Windows 路径分隔符替换掉）
        source_id = rel_path.replace("\\", "/")

        success, msg = process_one_file(
            recognizer,
            in_path=in_path,
            out_path=out_path,
            source_id=source_id,
            engine=engine,
            write_db=(not args.no_db)
        )
        if success:
            ok += 1
            print(f"✅ {rel_path} -> {os.path.relpath(out_path, out_root)}")
        else:
            fail += 1
            failures.append(msg)
            print(f"❌ {msg}")
            # 可打印更详细的堆栈用于调试：
            # traceback.print_exc()

    print("\n====== Summary ======")
    print(f"Input dir:  {in_root}")
    print(f"Output dir: {out_root}")
    print(f"Backend:    {args.backend}")
    print(f"DB write:   {'OFF' if args.no_db else 'ON'}")
    print(f"Processed:  {total}")
    print(f"Succeeded:  {ok}")
    print(f"Failed:     {fail}")
    if failures:
        print("\nFailures:")
        for m in failures[:20]:
            print(" -", m)
        if len(failures) > 20:
            print(f"... and {len(failures)-20} more")


if __name__ == "__main__":
    main()
