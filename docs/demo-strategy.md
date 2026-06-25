# SentiSync Demo Strategy

SentiSync is best presented as a reproducible local demo plus screenshots or a short walkthrough video. It should not advertise a public live demo unless the backend and extension dependencies are actually hosted and monitored.

## Current Public Demo

- README screenshot gallery:
  - `docs/examples/demo-summary.jpg`
  - `docs/examples/demo-trends.jpg`
  - `docs/examples/demo-wordcloud.jpg`
- Original full-resolution screenshots remain in `docs/examples/`.
- The README explains that the Flask API and Chrome extension must be configured locally.

## Why Not GitHub Pages

GitHub Pages only hosts static files. It cannot run:

- The Flask inference API.
- The LightGBM and TF-IDF model artifacts.
- YouTube Data API requests with a private API key.
- Chrome extension runtime behavior.
- Optional AWS deployment infrastructure.

For that reason, a GitHub Pages site would only be a static walkthrough. The README already handles that job well enough for now.

## Recommended Public Demo Format

The strongest next demo artifact would be a short GIF or video that shows:

1. Opening a YouTube video.
2. Launching the SentiSync extension.
3. Fetching comments.
4. Displaying summary metrics.
5. Scrolling through sentiment distribution, trend charts, word cloud, and top comments.

Keep the video around 30 to 60 seconds. It should show the product behavior, not the installation process.

## Local Reproduction Checklist

Before recording a walkthrough:

- Install API dependencies from `requirements-api.txt`.
- Download NLTK resources with `python -m nltk.downloader stopwords wordnet`.
- Start the backend with `python flask_app/app.py`.
- Copy `yt-chrome-plugin-frontend/config.js.example` to `yt-chrome-plugin-frontend/config.js`.
- Set `API_URL` to the backend URL.
- Set `API_KEY` to a valid YouTube Data API key.
- Load `yt-chrome-plugin-frontend/` as an unpacked Chrome extension.
- Open a YouTube video with enough comments to make the charts interesting.

## If A Hosted Demo Is Added Later

A real public live demo should only be linked after:

- The backend is deployed and health-checked.
- API keys are stored in server-side secrets, not the public repo.
- Rate limits and expected costs are understood.
- The extension or demo UI has a stable public backend URL.
- The README clearly labels the hosted demo as maintained.
