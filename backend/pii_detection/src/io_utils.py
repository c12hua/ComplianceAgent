# io_utils.py (修改版)
from __future__ import annotations
import os, json
from typing import Iterator, Tuple

def load_structured_from_dir(input_dir: str) -> Iterator[Tuple[str, str, list]]:
    """
    Yield (source_id, source_path, tokens).
    输入文件的每一行就是一个 JSON 数组（由上游分词/分句模型生成），
    我们这里直接读取，不再拼接成整段文本。
    
    示例行:
      ["患者","张某","，","男","，","45","岁"]
    """
    for root, _, files in os.walk(input_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(('.txt','.jsonl','.json')):
                continue
            path = os.path.join(root, fn)
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tokens = json.loads(line)
                        if not isinstance(tokens, list):
                            continue
                    except Exception:
                        continue
                    source_id = f"{os.path.relpath(path, input_dir)}__{i}"
                    yield source_id, path, tokens
