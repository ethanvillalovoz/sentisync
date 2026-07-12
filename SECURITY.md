# Security Policy

## Reporting Security Issues

Please do not open public issues for vulnerabilities, leaked credentials, or deployment secrets.

Email Ethan Villalovoz at `ethan.villalovoz@gmail.com` with:

- A short description of the issue.
- Steps to reproduce or affected files.
- Whether the issue affects the Flask API, Chrome extension, Docker image, or AWS deployment workflow.

## Secrets

Do not commit:

- YouTube Data API keys.
- AWS credentials.
- `.env` files.

The tracked extension config contains only a public API origin. Keep `YOUTUBE_API_KEY` on the Flask server.

## Sensitive Boundaries

- Configure allowed browser and extension origins with `SENTISYNC_ALLOWED_ORIGINS`.
- Treat `lgbm_model.pkl` and `tfidf_vectorizer.pkl` as trusted executable artifacts. Python pickle files can execute code while loading; do not replace them with unreviewed downloads.
- Do not include raw comments, user identifiers, cloud credentials, or production extension IDs in issues and screenshots.
