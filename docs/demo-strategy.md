# Demo Strategy

SentiSync has two public demonstration paths:

1. `popup.html?demo=1` runs a deterministic, credential-free product preview.
2. `docs/media/sentisync-demo.gif` shows the verified interaction directly in GitHub and portfolio surfaces.

Neither path claims to run the trained model. Live analysis requires the Flask service, trusted model artifacts, a server-side YouTube API key, and an extension origin listed in `SENTISYNC_ALLOWED_ORIGINS`.

When recording future demos, show the current video, loading pipeline, audience overview, and raw comment tab. Do not show API keys, browser extension IDs used in production, or private cloud infrastructure.
