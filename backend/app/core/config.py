import os
from pathlib import Path


def _load_dotenv() -> None:
    """Safely loads key-value pairs from .env without requiring third-party packages."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass

    backend_dir = Path(__file__).resolve().parent.parent.parent
    candidate_paths = [
        Path.cwd() / ".env",
        backend_dir / ".env",
        backend_dir.parent / ".env",
    ]
    for env_path in candidate_paths:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
            except Exception:
                pass
            break


_load_dotenv()


class Settings:
    PROJECT_NAME: str = "SentinelAI Security Gateway"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "true").lower() == "true"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TIMEOUT_SECONDS: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "15.0"))
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]


settings = Settings()
