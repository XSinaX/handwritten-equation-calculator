from shared.config import TOKENS, BLANK_TOKEN


class CTCTokenizer:
    def __init__(self):
        self.tokens = TOKENS
        self.blank_token = BLANK_TOKEN

        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

        self.blank_index = self.token_to_id[self.blank_token]

    def decode(self, token_ids):
        decoded = []

        for token_id in token_ids:
            token = self.tokens[int(token_id)]

            if token == self.blank_token:
                continue

            decoded.append(token)

        return "".join(decoded)


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


def decode_logits(logits):
    tokenizer = CTCTokenizer()

    predictions = greedy_decode(
        logits,
        tokenizer,
    )

    return predictions[0]
