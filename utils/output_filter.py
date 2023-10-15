from enum import Enum

class OutputFilter(Enum):
    zephyr = "<|assistant|>"
    mistral = ""
    llama = ""