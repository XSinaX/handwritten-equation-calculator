# Handwritten Equation Calculator

The project contains:

- `web` — Next.js frontend
- `api` — FastAPI backend
- `ml` — model training pipeline

## Web setup

```powershell
cd web
npm install
npm run dev
```

Open http://localhost:3000.

## API setup

Run from the project root:

```powershell
py -m venv api\api_venv
.\api\api_venv\Scripts\Activate.ps1
python -m pip install -r api\requirements.txt
python -m uvicorn app.main:app --reload
```

## ML setup

Run from the project root:

```powershell
py -m venv ml\ml_venv
.\ml\ml_venv\Scripts\Activate.ps1
python -m pip install -r ml\requirements.txt

```

### Train on the GPU (important)

`pip install torch` on Windows installs a **CPU-only** build, even if you have a
graphics card. Training then takes 1-2 days instead of under an hour.
`python -m ml.trainer` prints a large warning when this happens.

Check what you have:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`+cpu` / `False` means CPU only. Fix it for your GPU:

- **NVIDIA:** `python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128`
  (or pick the command for your CUDA version at https://pytorch.org/get-started/locally/)
- **AMD Radeon (e.g. RX 9070 XT) on Windows:** use AMD's ROCm build of PyTorch,
  following https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
  (select your GPU there; the RX 9070 XT is `gfx1201`). Use a fresh venv with the
  Python version AMD lists. If the PC also has a built-in Radeon GPU, the trainer
  automatically picks the card with the most memory.

Once `torch.cuda.is_available()` prints `True`, run the steps below.

```powershell
python -m ml.data_pipeline.data_cleaning      # once: builds the real-handwriting test set
python -m ml.data_pipeline.human_glyphs        # once: real symbol drawings cut from human equations
python -m ml.data_pipeline.equation_generator  # writes generated/train, generated/val, generated/test
python -m ml.data_pipeline.preprocessing
python -m ml.data_pipeline.image_symbols       # recommended: image-based equations (see below)
python -m ml.data_pipeline.photo_style         # recommended: equations as the API sees them after a photo
python -m ml.trainer                           # ends with test + real-handwriting scores
python -m ml.evaluate                          # re-score the saved model at any time
```

Choose the model combination with `--image-model` (cnn, resnet, vision_transformer)
and `--sequence-model` (bilstm, gru, transformer); the default is cnn + bilstm.
The model is saved as `ml/trained_models/<image>-<sequence>.pt`. All 9 combinations are
supported. To use one on the website, copy its `.pt` file into `api/models/`.

The trainer uses the most powerful GPU automatically (by compute units). To force one,
add `--gpu N` using the number shown in the "Found GPU N" lines.

```powershell
python -m ml.trainer --image-model cnn --sequence-model bilstm
python -m ml.evaluate --image-model cnn --sequence-model bilstm
```

Train, validation and test equations are built from **different** symbol
drawings, so the validation score reflects handwriting the model has not seen.
The "Real handwriting" score from `ml.evaluate` is the number to trust: it is
measured on human-written equations from MathWriting that the model never trains on.

## Training data sources

- **Ink-built equations** (`equation_generator.py`): built from MathWriting's ~30 handwritten
  drawings per symbol, plus ~1,400 real drawings cut from human-written equations
  (`human_glyphs.py`, MathWriting train split only; added to the training pool only).
- **Image-built equations** (`image_symbols.py`): built from the Xainano and sagyam image
  datasets (thousands of drawings per symbol). Put `Xainano_Raw/` and `sagyam_raw/` in
  `ml/data/`. The trainer adds `ml/data/image_generated/train` automatically when it exists.
  The first run cleans every symbol image and caches the result in
  `ml/data/image_symbol_bank.pkl` (a few minutes); later runs take about a minute.
- **Photo-style equations** (`photo_style.py`): ink-built training equations drawn as simulated
  phone photos (`photo_simulation.py`) and run through the API's own preprocessing
  (`shared/image_preprocessing.py`), so the model trains on exactly what it sees in use.
  The trainer adds `ml/data/photo_generated/train` automatically when it exists.
- **Real handwriting layout** (`layout_stats.py`): symbol sizes, vertical positions and gaps
  were measured from real handwritten equations (MathWriting train split only) and both
  generators sample from them. The measurements are committed in
  `ml/data_pipeline/layout_stats.json`; re-measure with `python -m ml.data_pipeline.layout_stats`
  only if you change how they are measured.

Equation rules: negative numbers are written with a plain `-` (`5+-2`, `5--2`, `6×-2`),
never with a `+` sign and never as a single number in brackets. A negative sign is written
close to its number, with a normal gap to the operator before it. The decoder only returns
equations that follow these rules (`shared/constrained_decoding.py`).

## Testing with photos (end to end through the API)

`api/tests/evaluate_api.py` sends every image in a folder through the real API
(upload -> preprocessing -> model -> decoding -> calculation) and scores the results.

1. Put your photos (JPG/PNG) in a folder, e.g. `my_photos/`.
2. Add `my_photos/labels.csv`:

    ```
    filename,equation
    photo1.jpg,12+3=
    photo2.jpg,5x-2
    photo3.jpg,(3+4)*2
    photo4.jpg,8/2
    ```

    Type equations naturally: `x` or `*` for times, `/` for divide, spaces are ignored.

3. Run from the project root:

    ```powershell
    python -m api.tests.evaluate_api --images my_photos
    python -m api.tests.evaluate_api --images my_photos --url http://localhost:8000   # against a running API
    ```

    Results for each image are saved to `my_photos/results.csv`.

To build a photo-like test set from real handwriting (paper texture, uneven lighting,
ruled lines, ink colours, slight rotation/perspective, blur, JPEG compression):

```powershell
python -m api.tests.make_photo_test_set --splits valid test
python -m api.tests.evaluate_api --images api/tests/photos
```

python -m ml.preprocessing.preprocess_image
