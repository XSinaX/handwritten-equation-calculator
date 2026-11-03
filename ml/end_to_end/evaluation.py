from ml.end_to_end.ctc.decoder import greedy_decode


def edit_distance(predicted_tokens, target_tokens):
    """
    Calculate the minimum number of insertions, deletions,
    and substitutions needed to transform the prediction
    into the target.
    """

    rows = len(target_tokens) + 1
    columns = len(predicted_tokens) + 1

    distances = []

    for row in range(rows):
        new_row = []

        for column in range(columns):
            new_row.append(0)

        distances.append(new_row)

    # Cost of deleting tokens
    for row in range(rows):
        distances[row][0] = row

    # Cost of inserting tokens
    for column in range(columns):
        distances[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):

            target_token = target_tokens[row - 1]
            predicted_token = predicted_tokens[column - 1]

            if target_token == predicted_token:
                substitution_cost = 0
            else:
                substitution_cost = 1

            deletion = distances[row - 1][column] + 1

            insertion = distances[row][column - 1] + 1

            substitution = distances[row - 1][column - 1] + substitution_cost

            distances[row][column] = min(
                deletion,
                insertion,
                substitution,
            )

    return distances[-1][-1]


def evaluate_batch(
    logits,
    labels,
    tokenizer,
):
    predictions = greedy_decode(
        logits,
        tokenizer,
    )

    correct_sequences = 0
    total_sequences = 0

    total_token_errors = 0
    total_target_tokens = 0

    for prediction, label in zip(
        predictions,
        labels,
    ):
        # Convert the target into the same canonical form
        # used by the model.
        target_tokens = tokenizer.encode(label)

        clean_target = tokenizer.decode(target_tokens)

        predicted_tokens = tokenizer.encode(prediction)

        # -------------------------
        # Sequence accuracy
        # -------------------------

        if prediction == clean_target:
            correct_sequences += 1

        total_sequences += 1

        # -------------------------
        # Token error rate
        # -------------------------

        errors = edit_distance(
            predicted_tokens,
            target_tokens,
        )

        total_token_errors += errors
        total_target_tokens += len(target_tokens)

    return (
        correct_sequences,
        total_sequences,
        total_token_errors,
        total_target_tokens,
    )
