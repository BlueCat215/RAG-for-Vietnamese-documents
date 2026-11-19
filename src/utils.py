import os
import yaml
import json
from typing import Dict, Any, List


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Đọc file cấu hình YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Không tìm thấy file config tại {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Đọc tệp jsonl trả về list các dictionary."""
    if not os.path.exists(file_path):
        return []
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
