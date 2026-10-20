import importlib.util
from functools import lru_cache
import sys
from .common import UPSTREAM

NORMALIZATION_VERSION = "hyporadise-english-plus-number-normalizer-v1"


@lru_cache(maxsize=1)
def whisper_module():
    source = UPSTREAM / "generate_data/whisper"
    if not source.is_dir():
        raise RuntimeError("Official source missing. Run scripts/fetch_upstream.py first.")
    sys.path.insert(0, str(source))
    import whisper
    if not str(whisper.__file__).startswith(str(source)):
        raise RuntimeError("Another whisper package was imported; use the pinned official fork.")
    return whisper


@lru_cache(maxsize=1)
def normalizer():
    whisper_module()
    from whisper.normalizers import EnglishTextNormalizer
    return EnglishTextNormalizer()


def normalize_text(text):
    engine = normalizer()
    # The fork comments out this existing number normalizer. Enable it here,
    # uniformly for candidates, labels, and predictions, and record the policy.
    return " ".join(engine.standardize_numbers(engine(text)).split())


@lru_cache(maxsize=1)
def prompter_class():
    path = UPSTREAM / "H2T-LoRA/utils/prompter.py"
    spec = importlib.util.spec_from_file_location("hyporadise_prompter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Prompter
