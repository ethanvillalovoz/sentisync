const SENTIMENT_META = {
  "1": { label: "Positive", className: "positive" },
  "0": { label: "Neutral", className: "neutral" },
  "-1": { label: "Negative", className: "negative" },
};

export function extractVideoId(rawUrl) {
  try {
    const url = new URL(rawUrl);
    const host = url.hostname.replace(/^www\./, "");

    if (host === "youtu.be") {
      const id = url.pathname.slice(1);
      return /^[\w-]{11}$/.test(id) ? id : null;
    }

    if (host !== "youtube.com" && host !== "m.youtube.com") {
      return null;
    }

    const id = url.searchParams.get("v");
    return id && /^[\w-]{11}$/.test(id) ? id : null;
  } catch {
    return null;
  }
}

export function getSentimentMeta(value) {
  return SENTIMENT_META[String(value)] ?? SENTIMENT_META["0"];
}

export function summarizePredictions(predictions) {
  const counts = { positive: 0, neutral: 0, negative: 0 };

  for (const item of predictions) {
    counts[getSentimentMeta(item.sentiment).className] += 1;
  }

  const total = predictions.length;
  const positivePercent = total ? Math.round((counts.positive / total) * 100) : 0;
  const mean = total
    ? predictions.reduce((sum, item) => sum + Number(item.sentiment || 0), 0) / total
    : 0;
  const score = ((Math.max(-1, Math.min(1, mean)) + 1) * 5).toFixed(1);

  return { counts, positivePercent, score, total };
}

export function createTrendSeries(predictions, pointCount = 8) {
  if (!predictions.length) return [];

  const bucketSize = Math.max(1, Math.ceil(predictions.length / pointCount));
  const points = [];

  for (let start = 0; start < predictions.length; start += bucketSize) {
    const bucket = predictions.slice(start, start + bucketSize);
    const average = bucket.reduce(
      (sum, item) => sum + Number(item.sentiment || 0),
      0,
    ) / bucket.length;
    points.push(Number(average.toFixed(3)));
  }

  return points;
}
