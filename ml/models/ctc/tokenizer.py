import torch
from shared.config import TOKENS, BLANK_TOKEN, COMMANDS


class CTCTokenizer:
    def __init__(self):
        self.tokens = TOKENS
        self.blank_token = BLANK_TOKEN

        self.token_to_id = {}

        for index, token in enumerate(self.tokens):
            self.token_to_id[token] = index

        self.blank_index = self.token_to_id[self.blank_token]
        self.num_classes = len(self.tokens)

    def encode(self, label):
        # Remove commands/characters that don't affect prediction
        label = label.replace("\\left", "")
        label = label.replace("\\right", "")
        label = label.replace(" ", "")
        label = label.replace("{", "")
        label = label.replace("}", "")

        encoded = []
        index = 0

        while index < len(label):
            command_found = False

            for command in COMMANDS:
                if label.startswith(command, index):
                    encoded.append(self.token_to_id[command])
                    index += len(command)
                    command_found = True
                    break

            if command_found:
                continue

            character = label[index]

            if character not in self.token_to_id:
                raise ValueError(f"Unsupported token {character!r} " f"in label {label!r}")

            encoded.append(self.token_to_id[character])
            index += 1

        return encoded

    def encode_batch(self, labels):
        encoded_labels = []
        lengths = []

        for label in labels:
            encoded = self.encode(label)
            # Flatten the encoded labels and store their lengths for CTC loss computation
            encoded_labels.extend(encoded)
            lengths.append(len(encoded))

        targets = torch.tensor(
            encoded_labels,
            dtype=torch.long,
        )

        target_lengths = torch.tensor(
            lengths,
            dtype=torch.long,
        )

        return targets, target_lengths

    def decode(self, token_ids):
        decoded = []

        for token_id in token_ids:
            token = self.tokens[int(token_id)]

            if token == self.blank_token:
                continue

            decoded.append(token)

        return "".join(decoded)
