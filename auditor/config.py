from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: SecretStr
    data_root: Path = Field(default_factory=lambda: Path.home() / "vlm-cnn-auditor-data")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3-vl:2b-instruct"
    yolo_weights: Path | None = None
    sample_hz: float = Field(default=1.0, ge=0.2, le=2.0)
    vlm_budget: int = Field(default=12, ge=1, le=12)
    frame_stride: int | None = Field(default=5, ge=5, le=5)
    request_timeout: float = Field(default=120.0, ge=1, le=600)
    ffprobe_binary: str = "ffprobe"
    server_port: int = Field(default=7860, ge=1024, le=65535)

    @field_validator("database_url")
    @classmethod
    def postgres_only(cls, value):
        if not value.get_secret_value().startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL must use postgresql+psycopg://")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def local_ollama_only(cls, value):
        url = urlsplit(value)
        if (
            url.scheme != "http"
            or url.hostname not in ("localhost", "127.0.0.1", "::1")
            or url.username
            or url.password
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            raise ValueError("OLLAMA_BASE_URL must be a loopback HTTP URL without credentials")
        return value.rstrip("/")

    @field_validator("data_root", "yolo_weights")
    @classmethod
    def absolute_paths(cls, value):
        return value.expanduser().resolve() if value is not None else None

    @property
    def weights_path(self) -> Path:
        return self.yolo_weights or self.data_root / "models" / "yolo11n.pt"

    def analysis_config(self) -> dict:
        return {
            **(
                {"frame_stride": self.frame_stride}
                if self.frame_stride is not None
                else {"sample_hz": self.sample_hz, "vlm_budget": self.vlm_budget}
            ),
            "yolo": {
                "model": self.weights_path.name,
                "class_id": 41,
                "conf": 0.25,
                "iou": 0.7,
                "imgsz": 640,
                "max_det": 100,
                "device": "cpu",
            },
            "qwen": {
                "model": self.ollama_model,
                "num_ctx": 8192,
                "num_predict": 1024,
                "temperature": 0,
                "image_long_edge": 1024,
                "timeout_seconds": self.request_timeout,
            },
            "prompt_version": "cup_regions_v1"
            if self.frame_stride is not None
            else "cup_review_v1",
            "rule_version": "cup_rules_v1",
            "schema_version": 1,
        }
