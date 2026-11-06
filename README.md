# Setup and Execution

Start each section from the repository root. Run the frontend and backend in separate terminals.

## Frontend

```powershell
cd web
npm ci
npm run dev
```

## Backend

```powershell
cd api
.\api_venv\Scripts\python.exe -m pip install -r requirements.txt
.\api_venv\Scripts\python.exe -m uvicorn api.app.main:app --app-dir .. --reload
```

## ML Dependencies

Both ML packages use the existing `ml_venv`.

```powershell
.\ml\ml_venv\Scripts\python.exe -m pip install -r ml/requirements.txt
```

## End-to-End ML

```powershell
# Prepare datasets
.\ml\ml_venv\Scripts\python.exe -m ml.data_pipeline.extract_datasets
.\ml\ml_venv\Scripts\python.exe -m ml.data_pipeline.equation_generator
.\ml\ml_venv\Scripts\python.exe -m ml.preprocessing --workers 6

# Train and evaluate
.\ml\ml_venv\Scripts\python.exe -m ml.trainer --model resnet-bilstm
```

## Segmentation-Based ML

```powershell
# Generate and render
.\ml\ml_venv\Scripts\python.exe -m ml_segmentation.data_pipeline.equation_generator --split all

# Preprocess
.\ml\ml_venv\Scripts\python.exe -m ml_segmentation.data_pipeline.preprocessing --split all --workers 6 --preview 0

# Segment and evaluate
.\ml\ml_venv\Scripts\python.exe -m ml_segmentation.segmentation.run --method baseline --split test

# Train and evaluate the merger
.\ml\ml_venv\Scripts\python.exe -m ml_segmentation.merger.run --segmentation baseline --merger geometry_mlp_simple --split all

# Evaluate the saved merger
.\ml\ml_venv\Scripts\python.exe -m ml_segmentation.merger.run --segmentation baseline --merger geometry_mlp_simple --split test

# Train and evaluate the classifier
.\ml\ml_venv\Scripts\python.exe -m ml_segmentation.classification.run --segmentation baseline --merger geometry_mlp_simple --model resnet18 --post all --split all

# Evaluate the saved classifier with all post-classification methods
.\ml\ml_venv\Scripts\python.exe -m ml_segmentation.classification.run --segmentation baseline --merger geometry_mlp_simple --model resnet18 --post all --split test
```
