"""Domain-specific exceptions with clean API mapping."""


class NeuralDocError(Exception):
    """Base error for all NeuralDoc failures."""


class UnsupportedFileTypeError(NeuralDocError):
    pass


class EmptyDocumentError(NeuralDocError):
    pass


class DocumentNotFoundError(NeuralDocError):
    pass


class IndexNotReadyError(NeuralDocError):
    pass


class EmbeddingError(NeuralDocError):
    pass
