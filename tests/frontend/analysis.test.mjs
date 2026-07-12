import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  createTrendSeries,
  extractVideoId,
  getSentimentMeta,
  summarizePredictions,
} from "../../yt-chrome-plugin-frontend/lib/analysis.js";

test("extractVideoId accepts canonical and short YouTube URLs", () => {
  assert.equal(extractVideoId("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ");
  assert.equal(extractVideoId("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ");
  assert.equal(extractVideoId("https://example.com/watch?v=dQw4w9WgXcQ"), null);
});

test("summarizePredictions reports distribution and a bounded score", () => {
  const summary = summarizePredictions([
    { sentiment: 1 },
    { sentiment: 1 },
    { sentiment: 0 },
    { sentiment: -1 },
  ]);

  assert.deepEqual(summary.counts, { positive: 2, neutral: 1, negative: 1 });
  assert.equal(summary.positivePercent, 50);
  assert.equal(summary.score, "6.3");
});

test("trend and unknown labels degrade safely", () => {
  assert.deepEqual(createTrendSeries([{ sentiment: 1 }, { sentiment: -1 }], 1), [0]);
  assert.equal(getSentimentMeta("unexpected").label, "Neutral");
});

test("extension surface avoids unsafe HTML rendering and broad permissions", async () => {
  const popupSource = await readFile(
    new URL("../../yt-chrome-plugin-frontend/popup.js", import.meta.url),
    "utf8",
  );
  const manifest = JSON.parse(
    await readFile(
      new URL("../../yt-chrome-plugin-frontend/manifest.json", import.meta.url),
      "utf8",
    ),
  );

  assert.equal(popupSource.includes("innerHTML"), false);
  assert.deepEqual(manifest.permissions, ["activeTab"]);
  assert.equal(manifest.host_permissions.includes("https://www.googleapis.com/*"), false);
});
