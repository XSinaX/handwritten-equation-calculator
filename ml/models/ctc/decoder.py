from shared.constrained_decoding import DEFAULT_BEAM_WIDTH, constrained_decode_batch


def greedy_decode(logits, tokenizer):
    predicted_ids = logits.argmax(dim=2)

    decoded_labels = []

    for sequence in predicted_ids:
        collapsed = []
        previous_id = None

        for token_id in sequence.tolist():

            # Remove CTC blank
            if token_id == tokenizer.blank_index:
                previous_id = token_id
                continue

            # Collapse consecutive duplicate predictions
            if token_id != previous_id:
                collapsed.append(token_id)

            previous_id = token_id

        decoded_label = tokenizer.decode(collapsed)
        decoded_labels.append(decoded_label)

    return decoded_labels


def constrained_decode(logits, tokenizer, beam_width=DEFAULT_BEAM_WIDTH):
    """
    Beam search that only returns valid equations (see
    shared/constrained_decoding.py). Falls back to the greedy result for an
    image when no valid equation can be found.
    """

    greedy = greedy_decode(logits, tokenizer)

    return constrained_decode_batch(
        logits,
        tokenizer.tokens,
        beam_width=beam_width,
        greedy_fallback=greedy,
    )
