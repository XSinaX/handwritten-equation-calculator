from ml.models.ctc.decoder import constrained_decode
from ml.models.ctc.tokenizer import CTCTokenizer

_tokenizer = CTCTokenizer()


def decode_logits(logits):
    """
    Turns the model output for one image into an equation string: the most
    likely reading that is a valid equation, or the plain greedy reading if
    no valid one is found (see shared/constrained_decoding.py).
    """

    return constrained_decode(logits, _tokenizer)[0]
