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
