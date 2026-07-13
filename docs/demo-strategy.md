# Demo Strategy

SentiSync has two public demonstration paths:

1. `popup.html?demo=1` runs a deterministic, credential-free product preview.
2. `docs/media/sentisync-demo.mp4` captures the extension at its native 390-pixel width and shows the loading, overview, and comments states directly in GitHub and portfolio surfaces.

`docs/media/sentisync-poster.webp` preserves the completed comment view for surfaces that do not play video.

Neither path claims to run the trained model. Live analysis requires the Flask service, trusted model artifacts, a server-side YouTube API key, and an extension origin listed in `SENTISYNC_ALLOWED_ORIGINS`.

When recording future demos, show the current video, loading pipeline, audience overview, and raw comment tab. Do not show API keys, browser extension IDs used in production, or private cloud infrastructure.
