"""Dataset loaders, processors, and FAISS retrieval module."""

from src.data.loaders import CaptionLoader, POPELoader, VQAv2Loader
from src.data.processors import AnnotationProcessor, CLIPEmbeddingExtractor, DataValidator, ImageProcessor
from src.data.retrieval import FAISSIndexBuilder, FAISSRetriever

__all__ = [
    "VQAv2Loader",
    "POPELoader",
    "CaptionLoader",
    "ImageProcessor",
    "AnnotationProcessor",
    "CLIPEmbeddingExtractor",
    "DataValidator",
    "FAISSIndexBuilder",
    "FAISSRetriever",
]
