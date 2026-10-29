"""
Scores the API on a folder of equation images (photos, scans, screenshots).

Every image goes through the real API endpoint - the same upload ->
preprocessing -> model -> decoding -> calculation path the website uses.

Folder layout:
    my_photos/
        labels.csv      columns: filename, equation   (or recognition_label)
        photo1.jpg      (images may also be in a my_photos/images/ subfolder)
        photo2.png
        ...

Equations in labels.csv can be typed naturally:
    12+3=   5x-2   8/2   (3+4)*2   10÷5   7×2
("x" or "*" = times, "/" = divide; spaces are ignored).

Run from the project root (in-process, no server needed):

    python -m api.tests.evaluate_api --images api/tests/photos

or against a running API server:

    python -m api.tests.evaluate_api --images my_photos --url http://localhost:8000

Results per image are saved to <images>/results.csv.
"""

import argparse
import csv
import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

from ml.models.ctc.tokenizer import CTCTokenizer
from shared.constrained_decoding import is_valid_equation

LENGTH_GROUPS = [(1, 4), (5, 9), (10, 14), (15, 99)]


def normalise(equation, tokenizer):
    """Converts a typed or LaTeX equation to the model's token string, or None."""

    text = (
        str(equation)
        .replace(" ", "")
        .replace("×", "\\times")
        .replace("*", "\\times")
        .replace("÷", "\\div")
        .replace("/", "\\div")
        .replace("−", "-")
    )

    # A plain "x" means times (but not inside "\times")
    text = text.replace("\\times", "\u0000").replace("x", "\\times").replace("X", "\\times").replace("\u0000", "\\times")

    try:
        return tokenizer.decode(tokenizer.encode(text))
    except ValueError:
        return None


def call_in_process(client, path, image_model, sequence_model):
    with open(path, "rb") as file:
        response = client.post(
            "/calculate/",
            files={"image": (path.name, file, mimetypes.guess_type(path.name)[0] or "image/jpeg")},
            data={"imageModel": image_model, "sequenceModel": sequence_model},
        )

    if response.status_code == 200:
        return response.json()["equation"], response.json()["answer"], ""

    return response.headers.get("X-Recognized-Equation", ""), None, response.json().get("detail", "")


def call_server(url, path, image_model, sequence_model):
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"

    body = b""
    for name, value in (("imageModel", image_model), ("sequenceModel", sequence_model)):
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()

    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        url.rstrip("/") + "/calculate/",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())
            return data["equation"], data["answer"], ""
    except urllib.error.HTTPError as error:
        detail = json.loads(error.read() or b"{}").get("detail", "")
        return error.headers.get("X-Recognized-Equation", ""), None, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="Folder with images and labels.csv")
    parser.add_argument("--url", help="Running API, e.g. http://localhost:8000 (default: in-process)")
    parser.add_argument("--image-model", default="cnn")
    parser.add_argument("--sequence-model", default="bilstm")
    parser.add_argument("--show", type=int, default=20, help="How many mistakes to print")
    arguments = parser.parse_args()

    tokenizer = CTCTokenizer()

    with open(arguments.images / "labels.csv", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if arguments.url:
        call = lambda path: call_server(arguments.url, path, arguments.image_model, arguments.sequence_model)
    else:
        from fastapi.testclient import TestClient

        from api.app.main import app

        client = TestClient(app)
        call = lambda path: call_in_process(client, path, arguments.image_model, arguments.sequence_model)

    results = []
    by_length = {group: [0, 0] for group in LENGTH_GROUPS}
    by_split = Counter()
    correct_by_split = Counter()
    valid_total = valid_correct = 0

    label_column = "equation" if rows and "equation" in rows[0] else "recognition_label"

    for row in rows:
        path = arguments.images / row["filename"]

        if not path.exists():
            path = arguments.images / "images" / row["filename"]

        expected = normalise(row[label_column], tokenizer)

        if expected is None or not path.exists():
            print(f"Skipping {row['filename']}: unsupported label or missing file")
            continue

        recognised, answer, error = call(path)
        correct = recognised == expected

        length = len(tokenizer.encode(expected))
        for group in LENGTH_GROUPS:
            if group[0] <= length <= group[1]:
                by_length[group][0] += correct
                by_length[group][1] += 1

        split = row.get("split", "")
        by_split[split] += 1
        correct_by_split[split] += correct

        if is_valid_equation([tokenizer.tokens[i] for i in tokenizer.encode(expected)]):
            valid_total += 1
            valid_correct += correct

        results.append(
            {
                "filename": row["filename"],
                "expected": expected,
                "recognised": recognised,
                "correct": correct,
                "answer": answer,
                "error": error,
            }
        )

    total = len(results)
    right = sum(r["correct"] for r in results)

    print(f"\nImages scored: {total}")
    print(f"Exact equation accuracy: {right}/{total} = {right / max(1, total) * 100:.1f}%")

    if valid_total < total:
        print(f"On labels that are valid equations ({valid_total}): {valid_correct / max(1, valid_total) * 100:.1f}%")

    print(
        "By length (symbols): "
        + " | ".join(
            f"{low}-{high if high < 99 else '+'}: {r / c * 100:.0f}% (n={c})"
            for (low, high), (r, c) in by_length.items()
            if c
        )
    )

    if len([s for s in by_split if s]) > 1:
        print("By split: " + " | ".join(f"{s}: {correct_by_split[s] / n * 100:.0f}% (n={n})" for s, n in by_split.items()))

    mistakes = [r for r in results if not r["correct"]]
    if mistakes:
        print(f"\nMistakes (expected | recognised){' - first ' + str(arguments.show) if len(mistakes) > arguments.show else ''}:")
        for r in mistakes[: arguments.show]:
            print(f"  {r['filename']:<28} {r['expected']:<26} | {r['recognised']}")

    with open(arguments.images / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()) if results else ["filename"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nPer-image results saved to {arguments.images / 'results.csv'}")


if __name__ == "__main__":
    main()
