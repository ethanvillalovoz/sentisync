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
- `yt-chrome-plugin-frontend/config.js`.

Use `yt-chrome-plugin-frontend/config.js.example` for local extension setup.
