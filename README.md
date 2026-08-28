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
```

Use `deactivate` to leave the environment.

## ML setup

Run from the project root:

```powershell
py -m venv ml\ml_venv
.\ml\ml_venv\Scripts\Activate.ps1
python -m pip install -r ml\requirements.txt
```

Use `deactivate` to leave the environment.
