from .units import LongUnitIndex, build_long_units
from .rerank import CrossEncoderReranker

__all__ = [
    "LongUnitIndex",
    "build_long_units",
    "CrossEncoderReranker",
    "LongRAGRetriever",
    "LongRAGSearchResult",
]


def __getattr__(name: str):
    if name in ("LongRAGRetriever", "LongRAGSearchResult"):
        from .retriever import LongRAGRetriever, LongRAGSearchResult

        return LongRAGRetriever if name == "LongRAGRetriever" else LongRAGSearchResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
