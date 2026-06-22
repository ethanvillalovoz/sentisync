# Contributing

Thanks for your interest in improving SentiSync. This project has three main surfaces: the Flask backend, the Chrome extension, and the DVC/MLflow model pipeline. Please keep changes focused and explain which surface they affect.

## Good Contribution Areas

- Backend API fixes, validation improvements, and endpoint tests.
- Chrome extension UX improvements that keep the same backend contract.
- DVC pipeline improvements for reproducible training and evaluation.
- Documentation, setup, and deployment clarifications.
- Small dependency or Docker improvements with verification.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-api.txt
python -m nltk.downloader stopwords wordnet
```

Install `requirements.txt` only when working on the DVC/MLflow training pipeline. Install `requirements-experiments.txt` only when running the historical experiment notebooks.

## Verification

Before opening a pull request, run:

```bash
python -m py_compile \
  flask_app/app.py \
  src/data/data_ingestion.py \
  src/data/data_preprocessing.py \
  src/model/model_building.py \
  src/model/model_evaluation.py \
  src/model/register_model.py

python -m unittest discover tests
docker build -t sentisync-backend .
```

If your change affects the Chrome extension, load `yt-chrome-plugin-frontend/` through `chrome://extensions` and test against a local or deployed backend.

## Pull Request Guidelines

- Keep each pull request scoped to one change.
- Include screenshots for extension UI changes.
- Update the README or docs when setup, API, deployment, or pipeline behavior changes.
- Add or update tests for deterministic backend behavior.
- Do not commit local `.env` files, `yt-chrome-plugin-frontend/config.js`, API keys, AWS credentials, or generated cache files.
- Follow the project code of conduct when participating in issues and pull requests.

## Reporting Issues

Please include:

- The command or workflow you ran.
- Backend, extension, Docker, or DVC context.
- Expected behavior.
- Actual behavior.
- Python version, browser version, and operating system.
- Screenshots, request payloads, or traceback output when helpful.
