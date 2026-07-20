# Figure contract: SentiSync system overview

## Communication job

By the end of the figure, a technical reader should understand that SentiSync keeps aggregate sentiment inspectable by linking a compact extension overview to the labeled comments behind it, while a separate offline pipeline produces the model used for live inference.

## Supported claim

The maintained system retrieves a bounded set of YouTube comments through a Flask API, transforms text with a stored trigram TF-IDF vectorizer, predicts three sentiment classes with a stored LightGBM model, and exposes distribution, temporal trend, and comment-level labels in the browser extension.

## Evidence used

- `flask_app/app.py` for the live API, validation limits, YouTube retrieval, and prediction interfaces.
- `src/data/`, `src/model/`, `dvc.yaml`, `dvc.lock`, and `params.yaml` for the offline Reddit-data training path and stored model artifacts.
- `yt-chrome-plugin-frontend/` for the extension workflow and deterministic demo fixtures.
- `media/overview-demo.png` and `media/comments-demo.png`, captured from `popup.html?demo=1`, for the product surfaces shown in the figure.

## Evidence boundary

- The product captures show bundled deterministic demo data, not a live YouTube request.
- The figure documents maintained code paths; it does not claim a deployed public service.
- Historical notebook figures are not used as current benchmark evidence.
- The Reddit-to-YouTube domain shift has not been evaluated, so the figure makes no accuracy or generalization claim.

## Visual hierarchy

1. The real extension overview is the dominant visual anchor.
2. The live inference rail explains how YouTube comments reach the extension.
3. A short link from the overview to a real comment-level excerpt demonstrates inspectability.
4. The offline training rail is visually separate and terminates at the stored artifacts used by inference.
5. The domain-shift caveat remains visible at README scale.

## Delivery formats

- Editable source: PowerPoint (`editable/sentisync-system-overview.pptx`)
- README export: SVG (`exports/sentisync-system-overview.svg`)
- Review export: PNG (`exports/sentisync-system-overview.png`)
- Print/export check: PDF (`exports/sentisync-system-overview.pdf`)
