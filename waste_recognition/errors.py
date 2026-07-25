"""Domain exceptions."""


class WasteRecognitionError(RuntimeError):
    """Base project error."""


class ConfigurationError(WasteRecognitionError):
    """Invalid configuration or incompatible options."""


class DatasetValidationError(WasteRecognitionError):
    """Invalid dataset structure or sample contents."""


class CheckpointError(WasteRecognitionError):
    """Invalid or incompatible model checkpoint."""
