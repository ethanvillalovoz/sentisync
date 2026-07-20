# SentiSync design direction

SentiSync is a Chrome extension, so it should feel native to a compact browser workflow. Its job is to move from the current YouTube video to a useful read of the discussion, then let the user inspect the comments behind each sentiment bucket.

## Principles

- Keep the surface at a realistic extension-popup width.
- Preserve the current video context through analysis and results.
- Make distribution rows actionable: selecting a sentiment opens the matching comments.
- Use a maintained chart library for the chronological trend rather than a decorative drawing.
- Keep demo mode and the local-model method explicit.
- Favor raw comments and model labels over generic audience-insight claims.
- Separate live YouTube inference from the offline Reddit-data training path in public diagrams.
- Keep the unevaluated Reddit-to-YouTube domain shift visible wherever the model pipeline is summarized.

## Avoid

- Recasting the extension as a full-page analytics dashboard.
- Invented creator metrics, engagement claims, or live YouTube data in demo mode.
- Decorative charts that cannot lead back to the underlying comments.
