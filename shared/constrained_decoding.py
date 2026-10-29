"""
Grammar-constrained CTC decoding.

Greedy decoding takes the single most likely symbol at every step, so it can
produce outputs that can't be a valid equation, e.g. "3+×4", "(2+5" or "+".
This decoder keeps several candidate readings at once (a CTC "prefix beam
search") and throws away any candidate that breaks the equation rules below.
The most likely *valid* reading wins. If no valid reading is found, the
caller should fall back to the greedy result.

Equation rules (matching what the generator produces and what people write):
  - a number cannot start with 0 unless it is just "0"
  - +, -, x and / need a number or ")" on their left
  - "-" can also be a negative sign: at the start, after "(", after "=" or
    after any operator ("5+-2", "5--2", "6x-2"); a sign is followed by a digit
  - "+" is never a sign (nobody writes "+6")
  - "(" can only follow the start, an operator or "="
  - ")" needs a matching "(", must follow a number or ")", and the brackets
    must contain an operator - a single number is never bracketed, "(5)"
  - "=" can only be the last symbol: people write "3+4=" and the app
    computes and shows the answer, so nothing is written after it
  - the equation must end with a number, ")" or "=", with all brackets closed

Shared by the training code (ml/) and the API (api/).
"""

import math
from collections import defaultdict

import numpy as np

from shared.config import BLANK_TOKEN

DIGITS = set("0123456789")
SIGN_OPERATORS = {"+", "-"}
OTHER_OPERATORS = {"\\times", "\\div"}

# What the previous token was, for the rules above
START, DIGIT, OPERATOR, SIGN, EQUALS, OPEN, CLOSE = range(7)

DEFAULT_BEAM_WIDTH = 10

# Symbols less likely than this at a time step are not considered there.
# Keeps the search fast without changing results in practice.
MIN_TOKEN_PROBABILITY = 1e-4

NEGATIVE_INFINITY = -math.inf


class GrammarState:
    __slots__ = ("previous", "groups", "equals_used", "number_is_zero")

    def __init__(self, previous=START, groups=(), equals_used=False, number_is_zero=False):
        self.previous = previous
        # One entry per open bracket: True once that group contains an operator
        self.groups = groups
        self.equals_used = equals_used
        # True when the current number so far is exactly "0"
        self.number_is_zero = number_is_zero


def _with_operator(groups):
    """Marks the innermost open bracket group as containing an operator."""

    if not groups:
        return groups

    return groups[:-1] + (True,)


def next_state(state, token):
    """
    Returns the state after adding token, or None if token isn't allowed here.
    """

    previous = state.previous

    # Nothing may follow "=" (the app writes the answer, not the user)
    if previous == EQUALS:
        return None

    if token in DIGITS:
        if previous == CLOSE:
            return None

        if previous == DIGIT and state.number_is_zero:
            return None  # no leading zeros like "05"

        starts_number = previous != DIGIT

        return GrammarState(
            DIGIT,
            state.groups,
            state.equals_used,
            number_is_zero=(token == "0") if starts_number else False,
        )

    if token == "-" and previous in (START, OPEN, EQUALS, OPERATOR):
        # Negative sign
        return GrammarState(SIGN, state.groups, state.equals_used)

    if token in SIGN_OPERATORS or token in OTHER_OPERATORS:
        if previous in (DIGIT, CLOSE):
            return GrammarState(OPERATOR, _with_operator(state.groups), state.equals_used)

        return None

    if token == "=":
        if state.equals_used or previous not in (DIGIT, CLOSE) or state.groups:
            return None

        return GrammarState(EQUALS, (), True)

    if token == "(":
        if previous in (START, OPERATOR, OPEN, EQUALS):
            return GrammarState(OPEN, state.groups + (False,), state.equals_used)

        return None

    if token == ")":
        # Must close a group that contains an operator: "(3+4)" not "(5)"
        if state.groups and state.groups[-1] and previous in (DIGIT, CLOSE):
            return GrammarState(CLOSE, state.groups[:-1], state.equals_used)

        return None

    return None


def is_complete(state):
    return not state.groups and state.previous in (DIGIT, CLOSE, EQUALS)


def is_valid_equation(tokens):
    """True if a list of tokens follows all the equation rules."""

    state = GrammarState()

    for token in tokens:
        state = next_state(state, token)

        if state is None:
            return False

    return is_complete(state)


def _log_add(a, b):
    if a == NEGATIVE_INFINITY:
        return b

    if b == NEGATIVE_INFINITY:
        return a

    if a > b:
        return a + math.log1p(math.exp(b - a))

    return b + math.log1p(math.exp(a - b))


def constrained_beam_search(log_probs, tokens, blank_index, beam_width=DEFAULT_BEAM_WIDTH):
    """
    CTC prefix beam search restricted to valid equations.

    Parameters
    ----------
    log_probs : np.ndarray [time, classes]
        Log-probabilities for one image.
    tokens : list[str]
        Token for each class index.

    Returns
    -------
    list[str] | None
        Best valid token sequence, or None if none was found.
    """

    min_log_probability = math.log(MIN_TOKEN_PROBABILITY)

    # prefix (tuple of class ids) -> [log P(ends in blank), log P(ends in symbol)]
    beams = {(): [0.0, NEGATIVE_INFINITY]}
    states = {(): GrammarState()}

    for step in log_probs:
        candidates = [
            index for index in np.nonzero(step > min_log_probability)[0].tolist() if index != blank_index
        ]

        blank_log_probability = float(step[blank_index])

        next_beams = defaultdict(lambda: [NEGATIVE_INFINITY, NEGATIVE_INFINITY])

        for prefix, (ends_blank, ends_symbol) in beams.items():
            total = _log_add(ends_blank, ends_symbol)

            # Stay on the same prefix by reading a blank
            entry = next_beams[prefix]
            entry[0] = _log_add(entry[0], total + blank_log_probability)

            # Stay on the same prefix by repeating its last symbol
            if prefix:
                last = prefix[-1]
                entry[1] = _log_add(entry[1], ends_symbol + float(step[last]))

            state = states[prefix]

            for index in candidates:
                new_state = next_state(state, tokens[index])

                if new_state is None:
                    continue

                new_prefix = prefix + (index,)
                symbol_log_probability = float(step[index])

                new_entry = next_beams[new_prefix]

                if prefix and prefix[-1] == index:
                    # The same symbol twice in a row needs a blank between them
                    new_entry[1] = _log_add(new_entry[1], ends_blank + symbol_log_probability)
                else:
                    new_entry[1] = _log_add(new_entry[1], total + symbol_log_probability)

                states[new_prefix] = new_state

        ranked = sorted(
            next_beams.items(),
            key=lambda item: _log_add(item[1][0], item[1][1]),
            reverse=True,
        )

        beams = dict(ranked[:beam_width])

    complete = [
        (prefix, _log_add(*scores)) for prefix, scores in beams.items() if prefix and is_complete(states[prefix])
    ]

    if not complete:
        return None

    best_prefix = max(complete, key=lambda item: item[1])[0]

    return [tokens[index] for index in best_prefix]


def constrained_decode_batch(logits, tokens, beam_width=DEFAULT_BEAM_WIDTH, greedy_fallback=None):
    """
    Decodes a batch of model outputs [batch, time, classes] into strings.

    greedy_fallback: list of greedy strings used when no valid equation
    is found for an image (keeps behaviour no worse than greedy).
    """

    blank_index = tokens.index(BLANK_TOKEN)

    log_probs = logits.float().log_softmax(dim=2).cpu().numpy()

    results = []

    for index, sample in enumerate(log_probs):
        best = constrained_beam_search(sample, tokens, blank_index, beam_width)

        if best is None:
            results.append(greedy_fallback[index] if greedy_fallback is not None else "")
        else:
            results.append("".join(best))

    return results
