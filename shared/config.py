# -------------------------
# Image dimensions
# -------------------------

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# -------------------------
# Tokenizer
# -------------------------

BLANK_TOKEN = "<blank>"

TOKENS = [
    BLANK_TOKEN,
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "+",
    "-",
    "=",
    "(",
    ")",
    ".",
    "\\div",
    "\\times",
]

COMMANDS = []

for token in TOKENS:
    if token.startswith("\\"):
        COMMANDS.append(token)

NUM_CLASSES = len(TOKENS)
