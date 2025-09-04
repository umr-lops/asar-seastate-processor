from importlib.metadata import version

try:
    __version__ = version("asar_seastate_processor")
except Exception:
    __version__ = "9999"
