import {
  createTrendSeries,
  extractVideoId,
  getSentimentMeta,
  summarizePredictions,
} from "./lib/analysis.js";
import { DEMO_PREDICTIONS, DEMO_VIDEO } from "./data/demo.js";

const config = globalThis.SENTISYNC_CONFIG ?? {
  API_URL: "http://localhost:8080",
  DEMO_MODE: false,
};
const query = new URLSearchParams(globalThis.location.search);
const canReadChromeTab = Boolean(globalThis.chrome?.tabs?.query);
const demoMode = config.DEMO_MODE || query.get("demo") === "1" || !canReadChromeTab;

const elements = {
  analyzeButton: document.querySelector("#analyze-button"),
  commentsPanel: document.querySelector("#comments-panel"),
  commentsTab: document.querySelector("#comments-tab"),
  commentList: document.querySelector("#comment-list"),
  distribution: document.querySelector("#distribution"),
  distributionCount: document.querySelector("#distribution-count"),
  footerStatus: document.querySelector("#footer-status"),
  introView: document.querySelector("#intro-view"),
  loadingView: document.querySelector("#loading-view"),
  metricPositive: document.querySelector("#metric-positive"),
  metricScore: document.querySelector("#metric-score"),
  metricTotal: document.querySelector("#metric-total"),
  modeBadge: document.querySelector("#mode-badge"),
  overviewPanel: document.querySelector("#overview-panel"),
  overviewTab: document.querySelector("#overview-tab"),
  pipelineItems: [...document.querySelectorAll("#pipeline-list li")],
  reanalyzeButton: document.querySelector("#reanalyze-button"),
  resultsView: document.querySelector("#results-view"),
  resultVideoTitle: document.querySelector("#result-video-title"),
  statusCopy: document.querySelector("#status-copy"),
  trendChart: document.querySelector("#trend-chart"),
  videoTitle: document.querySelector("#video-title"),
};

let activeVideo = null;

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setView(view) {
  elements.introView.hidden = view !== "intro";
  elements.loadingView.hidden = view !== "loading";
  elements.resultsView.hidden = view !== "results";
}

function setStatus(message, { error = false } = {}) {
  elements.statusCopy.textContent = message;
  elements.statusCopy.classList.toggle("is-error", error);
}

function getActiveChromeTab() {
  return new Promise((resolve, reject) => {
    globalThis.chrome.tabs.query(
      { active: true, currentWindow: true },
      (tabs) => {
        const runtimeError = globalThis.chrome.runtime?.lastError;
        if (runtimeError) {
          reject(new Error(runtimeError.message));
          return;
        }
        resolve(tabs[0] ?? null);
      },
    );
  });
}

async function detectVideo() {
  elements.modeBadge.textContent = demoMode ? "Demo" : "Local";

  if (demoMode) {
    activeVideo = DEMO_VIDEO;
    elements.videoTitle.textContent = DEMO_VIDEO.title;
    elements.analyzeButton.disabled = false;
    setStatus("Preview uses bundled comments and deterministic model output.");
    return;
  }

  try {
    const tab = await getActiveChromeTab();
    const videoId = extractVideoId(tab?.url ?? "");
    if (!videoId) {
      elements.videoTitle.textContent = "Open a YouTube video to begin";
      setStatus("SentiSync only reads standard YouTube watch pages.", { error: true });
      return;
    }

    activeVideo = { id: videoId, title: tab.title || `YouTube video ${videoId}` };
    elements.videoTitle.textContent = activeVideo.title;
    elements.analyzeButton.disabled = false;
    setStatus("Ready to retrieve and classify up to 500 top-level comments.");
  } catch {
    elements.videoTitle.textContent = "Unable to read the active tab";
    setStatus("Check the extension's active-tab permission and try again.", { error: true });
  }
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${config.API_URL.replace(/\/$/, "")}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

async function runLiveAnalysis() {
  const commentsResponse = await requestJson("/youtube/comments", {
    method: "POST",
    body: JSON.stringify({ video_id: activeVideo.id, max_results: 500 }),
  });
  return requestJson("/predict_with_timestamps", {
    method: "POST",
    body: JSON.stringify({ comments: commentsResponse.comments }),
  });
}

async function animatePipeline(promise) {
  for (let index = 0; index < elements.pipelineItems.length; index += 1) {
    elements.pipelineItems.forEach((item, itemIndex) => {
      item.classList.toggle("is-active", itemIndex <= index);
    });
    await new Promise((resolve) => setTimeout(resolve, 180));
  }
  return promise;
}

function renderDistribution(summary) {
  const rows = [
    ["Positive", "positive", summary.counts.positive],
    ["Neutral", "neutral", summary.counts.neutral],
    ["Negative", "negative", summary.counts.negative],
  ];
  const fragment = document.createDocumentFragment();

  for (const [label, className, count] of rows) {
    const percentage = summary.total ? Math.round((count / summary.total) * 100) : 0;
    const row = createElement("div", "distribution-row");
    const track = createElement("div", "bar-track");
    const fill = createElement("div", `bar-fill ${className}`);
    fill.style.width = `${percentage}%`;
    track.append(fill);
    row.append(
      createElement("span", null, label),
      track,
      createElement("span", "distribution-value", `${percentage}%`),
    );
    fragment.append(row);
  }

  elements.distribution.replaceChildren(fragment);
  elements.distributionCount.textContent = `${summary.total} comments`;
}

function renderTrend(predictions) {
  const values = createTrendSeries(predictions);
  const width = 330;
  const height = 92;
  const inset = 5;
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Average sentiment from oldest to newest comments");

  for (const y of [inset, height / 2, height - inset]) {
    const guide = document.createElementNS(namespace, "line");
    guide.setAttribute("x1", "0");
    guide.setAttribute("x2", String(width));
    guide.setAttribute("y1", String(y));
    guide.setAttribute("y2", String(y));
    guide.setAttribute("stroke", "#ded9d2");
    guide.setAttribute("stroke-width", "1");
    svg.append(guide);
  }

  const points = values.map((value, index) => {
    const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
    const y = inset + ((1 - value) / 2) * (height - inset * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const line = document.createElementNS(namespace, "polyline");
  line.setAttribute("points", points.join(" "));
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "#0f0f0f");
  line.setAttribute("stroke-width", "2");
  line.setAttribute("vector-effect", "non-scaling-stroke");
  svg.append(line);

  elements.trendChart.replaceChildren(svg);
}

function renderComments(predictions) {
  const fragment = document.createDocumentFragment();
  predictions.slice(0, 12).forEach((item, index) => {
    const sentiment = getSentimentMeta(item.sentiment);
    const row = createElement("li", "comment-item");
    row.append(
      createElement("span", "comment-index", String(index + 1).padStart(2, "0")),
      createElement("p", "comment-text", item.comment),
      createElement("span", `comment-label ${sentiment.className}`, sentiment.label),
    );
    fragment.append(row);
  });
  elements.commentList.replaceChildren(fragment);
}

function renderResults(predictions) {
  const summary = summarizePredictions(predictions);
  elements.resultVideoTitle.textContent = activeVideo.title;
  elements.metricTotal.textContent = String(summary.total);
  elements.metricPositive.textContent = `${summary.positivePercent}%`;
  elements.metricScore.textContent = summary.score;
  renderDistribution(summary);
  renderTrend(predictions);
  renderComments(predictions);
  elements.footerStatus.textContent = demoMode ? "Demo data" : "Live analysis";
  setView("results");
}

async function analyze() {
  if (!activeVideo) return;
  elements.pipelineItems.forEach((item) => item.classList.remove("is-active"));
  elements.footerStatus.textContent = "Analyzing";
  setView("loading");

  try {
    const analysisPromise = demoMode
      ? Promise.resolve(DEMO_PREDICTIONS)
      : runLiveAnalysis();
    const predictions = await animatePipeline(analysisPromise);
    if (!Array.isArray(predictions) || predictions.length === 0) {
      throw new Error("No comments were returned for analysis.");
    }
    renderResults(predictions);
  } catch (error) {
    setView("intro");
    setStatus(error.message || "Analysis failed. Check the API and try again.", { error: true });
    elements.footerStatus.textContent = "Unavailable";
  }
}

function selectTab(name) {
  const overviewActive = name === "overview";
  elements.overviewTab.classList.toggle("is-active", overviewActive);
  elements.overviewTab.setAttribute("aria-selected", String(overviewActive));
  elements.commentsTab.classList.toggle("is-active", !overviewActive);
  elements.commentsTab.setAttribute("aria-selected", String(!overviewActive));
  elements.overviewPanel.hidden = !overviewActive;
  elements.commentsPanel.hidden = overviewActive;
}

elements.analyzeButton.addEventListener("click", analyze);
elements.reanalyzeButton.addEventListener("click", analyze);
elements.overviewTab.addEventListener("click", () => selectTab("overview"));
elements.commentsTab.addEventListener("click", () => selectTab("comments"));

detectVideo();
