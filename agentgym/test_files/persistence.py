import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from agentgym.test_files.models import EnvInfo, ToolExecutionRecord


class EnvPersistence:

    def __init__(self, base_dir: str = "env_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def _get_env_filename(self, env_id: str, timestamp: Optional[int] = None) -> str:
        if timestamp:
            dt = datetime.fromtimestamp(timestamp / 1000)
        else:
            dt = datetime.now()

        date_str = dt.strftime("%Y-%m-%d")
        return f"{date_str}-{env_id}.json"

    def _get_env_path(self, env_id: str, timestamp: Optional[int] = None) -> Path:
        filename = self._get_env_filename(env_id, timestamp)
        return self.base_dir / filename

    def save_env(self, env_info: EnvInfo, tool_history: List[ToolExecutionRecord],
                init_timestamp: Optional[int] = None, close_timestamp: Optional[int] = None) -> str:
        env_data = {
            "env_info": env_info.model_dump(),
            "tool_history": [record.model_dump() for record in tool_history],
            "init_timestamp": init_timestamp,
            "close_timestamp": close_timestamp,
            "saved_at": int(datetime.now().timestamp() * 1000)
        }

        timestamp_for_filename = init_timestamp or env_info.created
        file_path = self._get_env_path(env_info.id, timestamp_for_filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(env_data, f, indent=2, ensure_ascii=False)

        return str(file_path)

    def load_env(self, env_id: str, timestamp: Optional[int] = None) -> Optional[Dict[str, Any]]:
        file_path = self._get_env_path(env_id, timestamp)

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None