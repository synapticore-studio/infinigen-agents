# Model Dependencies - Lokale distilierte Modelle
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_ai import Agent

# Simple dependency injection

logger = logging.getLogger(__name__)


@dataclass
class LocalModelConfig:
    """Konfiguration für lokale distilierte Modelle"""

    model_type: str = "ollama"  # "ollama", "transformers", "llama_cpp"
    model_path: str = "./models/infinigen-distilled"
    ollama_model: str = "infinigen:latest"
    ollama_base_url: str = "http://localhost:11434"
    max_tokens: int = 4000
    temperature: float = 0.7
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Modell-spezifische Einstellungen
    context_length: int = 8192
    batch_size: int = 1
    num_threads: int = 4


class LocalModelProvider:
    """Provider für lokale distilierte Modelle"""

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self.model = None
        self.agent = None
        self.logger = logging.getLogger(__name__)
        self._initialize_model()

    def _initialize_model(self):
        """Initialisiere das lokale Modell"""
        try:
            if self.config.model_type == "ollama":
                self._init_ollama()
            elif self.config.model_type == "transformers":
                self._init_transformers()
            elif self.config.model_type == "llama_cpp":
                self._init_llama_cpp()
            else:
                raise ValueError(f"Unbekannter Modell-Typ: {self.config.model_type}")

            self.logger.info(
                f"✅ Lokales Modell initialisiert: {self.config.model_type}"
            )

        except Exception as e:
            self.logger.error(f"❌ Fehler bei Modell-Initialisierung: {e}")
            self.model = None

    def _init_ollama(self):
        """Initialisiere Ollama-Modell"""
        try:
            import requests

            # Teste Ollama-Verbindung
            response = requests.get(
                f"{self.config.ollama_base_url}/api/tags", timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                if self.config.ollama_model in model_names:
                    # Erstelle einfachen Agent für Ollama
                    self.agent = Agent(
                        model=self.config.ollama_model,
                        base_url=f"{self.config.ollama_base_url}/v1",
                        api_key="ollama",
                    )
                    self.logger.info(
                        f"✅ Ollama-Modell gefunden: {self.config.ollama_model}"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ Ollama-Modell nicht gefunden: {self.config.ollama_model}"
                    )
                    self.logger.info(f"Verfügbare Modelle: {model_names}")
                    # Fallback auf erstes verfügbares Modell
                    if model_names:
                        self.agent = Agent(
                            model=model_names[0],
                            base_url=f"{self.config.ollama_base_url}/v1",
                            api_key="ollama",
                        )
                        self.logger.info(
                            f"🔄 Verwende Fallback-Modell: {model_names[0]}"
                        )
            else:
                self.logger.error(
                    f"❌ Ollama-Server nicht erreichbar: {response.status_code}"
                )

        except ImportError:
            self.logger.error("❌ requests nicht installiert - Ollama nicht verfügbar")
        except Exception as e:
            self.logger.error(f"❌ Ollama-Initialisierung fehlgeschlagen: {e}")

    def _init_transformers(self):
        """Initialisiere Transformers-Modell - Fallback auf lokales Modell"""
        try:
            # Für Transformers verwenden wir ein lokales Modell
            self.agent = Agent(
                model="local-transformers",
                base_url="http://localhost:8000/v1",  # Lokale API
                api_key="local",
            )
            self.logger.info("✅ Transformers-Modell konfiguriert (lokale API)")

        except Exception as e:
            self.logger.error(f"❌ Transformers-Initialisierung fehlgeschlagen: {e}")

    def _init_llama_cpp(self):
        """Initialisiere llama.cpp-Modell - Fallback auf lokales Modell"""
        try:
            # Für llama.cpp verwenden wir ein lokales Modell
            self.agent = Agent(
                model="local-llama-cpp",
                base_url="http://localhost:8080/v1",  # Lokale API
                api_key="local",
            )
            self.logger.info("✅ llama.cpp-Modell konfiguriert (lokale API)")

        except Exception as e:
            self.logger.error(f"❌ llama.cpp-Initialisierung fehlgeschlagen: {e}")


def get_local_model_config() -> LocalModelConfig:
    """Hole lokale Modell-Konfiguration"""
    return LocalModelConfig()


def get_model_provider() -> LocalModelProvider:
    """Hole Modell-Provider"""
    config = get_local_model_config()
    return LocalModelProvider(config)


# Dependency für Agents
ModelProviderDep = get_model_provider
