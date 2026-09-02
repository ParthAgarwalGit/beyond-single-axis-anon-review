import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_frozen_config


@pytest.fixture(scope="session")
def cfg():
    config, _ = load_frozen_config()
    return config


def char_tokenize(text):
    """Deterministic stand-in tokenizer for CPU-only tests: one token per
    character. Prefix-of-text always tokenizes to prefix-of-ids, matching
    the property the frozen boundary check relies on."""
    return [ord(c) for c in text]
