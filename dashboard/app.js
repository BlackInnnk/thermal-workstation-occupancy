const workstation = {
  name: "Soldering Workstation",
  occupied: false,
  displayStatus: "OFFLINE",
  confidence: null,
  peak: null,
  safety: "UNAVAILABLE",
  detectionMode: null,
  lastChanged: null,
  history: Array(12).fill(null),
};

let connectionState = "connecting";
let liveConnected = false;
let lastSensorTimestamp = null;
let lastSnapshotTimestamp = null;
let lastSnapshotRefreshAt = 0;
let snapshotIntervalSeconds = 30;
let thermalSnapshotUrl = "/data/runtime/thermal_view.jpg";
let pollInFlight = false;
let eventLog = [{ time: Date.now(), message: "Waiting for the live sensor feed" }];

const canvas = document.getElementById("heatmapCanvas");
const ctx = canvas.getContext("2d");

const ids = {
  stateAnnouncement: document.getElementById("stateAnnouncement"),
  lastUpdated: document.getElementById("lastUpdated"),
  frameLabel: document.getElementById("frameLabel"),
  eventCount: document.getElementById("eventCount"),
  eventLog: document.getElementById("eventLog"),
  syncLabel: document.getElementById("syncLabel"),
  syncDot: document.getElementById("syncDot"),
  detectionMode: document.getElementById("detectionMode"),
  snapshotStatus: document.getElementById("snapshotStatus"),
  stackSnapshotInterval: document.getElementById("stackSnapshotInterval"),
  thermalCaption: document.getElementById("thermalCaption"),
  connectionState: document.getElementById("connectionState"),
  thermalSnapshot: document.getElementById("thermalSnapshot"),
  feedPlaceholder: document.getElementById("feedPlaceholder"),
  feedPlaceholderTitle: document.getElementById("feedPlaceholderTitle"),
  feedPlaceholderText: document.getElementById("feedPlaceholderText"),
};

const OCCUPANCY_STATES = new Set(["FREE", "OCCUPIED", "RECENTLY_USED"]);
const SAFETY_STATES = new Set(["SAFE", "IN_USE", "MONITORING", "COOLING", "UNATTENDED_HOT"]);

function formatTime(date) {
  return new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function timeAgo(timestamp) {
  if (!Number.isFinite(timestamp)) return "--";
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 2) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

function addEvent(message) {
  eventLog.unshift({ time: Date.now(), message });
  eventLog = eventLog.slice(0, 8);
}

function titleCaseState(value) {
  return String(value || "--")
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function isWarningSafety(safety) {
  return safety === "UNATTENDED_HOT" || safety === "MONITORING";
}

function safetyClass(safety) {
  if (safety === "UNATTENDED_HOT") return "is-danger";
  if (safety === "COOLING") return "is-cooling";
  if (safety === "MONITORING") return "is-monitoring";
  if (safety === "IN_USE") return "is-in-use";
  if (safety === "SAFE") return "is-safe";
  return "is-offline";
}

function normaliseRuntimeUrl(url, fallback) {
  if (!url) return fallback;
  if (url.startsWith("../data/runtime/")) return url.replace("../data/runtime/", "/data/runtime/");
  if (url.startsWith("data/runtime/")) return `/${url}`;
  return url;
}

function liveStateChangedAt(payload) {
  const explicitChangedAt = Date.parse(payload?.occupancy?.changed_at || "");
  if (Number.isFinite(explicitChangedAt)) return explicitChangedAt;

  const stateSeconds = Number(payload?.occupancy?.state_seconds);
  if (!Number.isFinite(stateSeconds)) return null;

  const statusTimestamp = Date.parse(payload?.timestamp || "");
  const baseTime = Number.isFinite(statusTimestamp) ? statusTimestamp : Date.now();
  return baseTime - stateSeconds * 1000;
}

function setConnectionState(nextState, message) {
  const changed = connectionState !== nextState;
  if (changed && message) {
    addEvent(message);
    ids.stateAnnouncement.textContent = message;
  }
  connectionState = nextState;
  liveConnected = nextState === "live";

  if (liveConnected) return;

  workstation.occupied = false;
  workstation.displayStatus = "OFFLINE";
  workstation.confidence = null;
  workstation.peak = null;
  workstation.safety = "UNAVAILABLE";
  workstation.detectionMode = null;
  workstation.lastChanged = null;
  workstation.history = Array(12).fill(null);
  ids.thermalSnapshot.classList.remove("is-visible");
  ids.feedPlaceholder.hidden = false;
}

function applyLiveStatus(payload) {
  const occupancyState = payload?.occupancy?.state;
  const safetyState = payload?.safety?.state;
  const statusTimestamp = Date.parse(payload?.timestamp || "");
  if (
    !OCCUPANCY_STATES.has(occupancyState)
    || !SAFETY_STATES.has(safetyState)
    || !Number.isFinite(statusTimestamp)
  ) {
    throw new Error("Incomplete live status payload");
  }

  const occupied = occupancyState === "OCCUPIED";
  const modelProbability = Number(payload?.model?.occupied_probability);
  const toolTemperature = Number(
    payload?.safety?.tool_temperature_c ?? payload?.metrics?.tool_p95_c,
  );
  const changedAt = liveStateChangedAt(payload);
  const snapshotTimestamp = Date.parse(payload?.snapshot?.updated_at || "");
  const configuredSnapshotInterval = Number(payload?.snapshot?.interval_seconds);
  const previousStatus = workstation.displayStatus;
  const previousSafety = workstation.safety;

  thermalSnapshotUrl = normaliseRuntimeUrl(payload?.snapshot?.url, thermalSnapshotUrl);
  if (Number.isFinite(configuredSnapshotInterval) && configuredSnapshotInterval > 0) {
    snapshotIntervalSeconds = configuredSnapshotInterval;
  }
  lastSensorTimestamp = statusTimestamp;
  setConnectionState("live", "Live sensor feed connected");

  if (previousStatus !== occupancyState || previousSafety !== safetyState) {
    const stateMessage =
      `Live: ${workstation.name} ${titleCaseState(occupancyState)}, safety ${titleCaseState(safetyState)}`;
    addEvent(stateMessage);
    ids.stateAnnouncement.textContent = stateMessage;
  }

  workstation.occupied = occupied;
  workstation.displayStatus = occupancyState;
  workstation.confidence = Number.isFinite(modelProbability)
    ? Math.round(modelProbability * 100)
    : null;
  workstation.peak = Number.isFinite(toolTemperature) ? toolTemperature : null;
  workstation.safety = safetyState;
  workstation.detectionMode = payload?.model ? "ML + rules" : "ROI rules";
  workstation.lastChanged = changedAt;
  workstation.history.push(occupied);
  workstation.history = workstation.history.slice(-12);

  if (Number.isFinite(snapshotTimestamp) && snapshotTimestamp !== lastSnapshotTimestamp) {
    lastSnapshotTimestamp = snapshotTimestamp;
    refreshThermalSnapshot(true);
  }
}

async function pollLiveStatus() {
  if (pollInFlight) return;
  pollInFlight = true;

  try {
    const cacheBust = Date.now();
    const [statusResponse, healthResponse] = await Promise.all([
      fetch(`/data/runtime/status.json?ts=${cacheBust}`, { cache: "no-store" }),
      fetch(`/healthz?ts=${cacheBust}`, { cache: "no-store" }),
    ]);
    if (!statusResponse.ok || !healthResponse.ok) throw new Error("Live endpoint unavailable");

    const [payload, health] = await Promise.all([
      statusResponse.json(),
      healthResponse.json(),
    ]);
    if (health?.sensor !== "fresh") {
      setConnectionState("stale", "Sensor status stopped updating");
      return;
    }

    applyLiveStatus(payload);
  } catch {
    setConnectionState("offline", "Live sensor feed unavailable");
  } finally {
    pollInFlight = false;
    render();
  }
}

function refreshThermalSnapshot(force = false) {
  if (!liveConnected) return;
  const now = Date.now();
  if (!force && now - lastSnapshotRefreshAt < snapshotIntervalSeconds * 1000) return;

  lastSnapshotRefreshAt = now;
  ids.thermalSnapshot.src = `${thermalSnapshotUrl}?ts=${lastSnapshotTimestamp || now}`;
  ids.frameLabel.textContent = Number.isFinite(lastSnapshotTimestamp)
    ? `Snapshot ${formatTime(new Date(lastSnapshotTimestamp))}`
    : "Snapshot pending";
  ids.snapshotStatus.textContent = "Refreshing";
}

function renderWorkstation() {
  const panel = document.getElementById("workstation-panel");
  const dot = document.getElementById("workstation-dot");
  const status = document.getElementById("workstation-status");
  const confidence = document.getElementById("workstation-confidence");
  const peak = document.getElementById("workstation-peak");
  const change = document.getElementById("workstation-change");
  const safety = document.getElementById("workstation-safety");
  const safetyPrimary = document.getElementById("workstation-safety-primary");
  const safetyCard = document.getElementById("workstation-safety-card");
  const safetyDot = document.getElementById("workstation-safety-dot");
  const safetyStateClass = safetyClass(workstation.safety);

  panel.classList.toggle("is-occupied", workstation.occupied);
  panel.classList.toggle("is-free", workstation.displayStatus === "FREE");
  panel.classList.toggle("is-recent", workstation.displayStatus === "RECENTLY_USED");
  panel.classList.toggle("is-offline", !liveConnected);
  panel.classList.toggle("is-warning", !workstation.occupied && isWarningSafety(workstation.safety));
  panel.classList.toggle("is-danger", workstation.safety === "UNATTENDED_HOT");

  dot.className = "status-indicator";
  if (!liveConnected) dot.classList.add("is-offline");
  else if (workstation.occupied) dot.classList.add("is-occupied");
  else if (workstation.displayStatus === "RECENTLY_USED") dot.classList.add("is-recent");
  else dot.classList.add("is-free");

  safetyCard.className = `state-card state-card-safety ${safetyStateClass}`;
  safetyDot.className = `status-indicator safety-indicator ${safetyStateClass}`;

  status.textContent = liveConnected ? titleCaseState(workstation.displayStatus) : "Offline";
  safetyPrimary.textContent = liveConnected ? titleCaseState(workstation.safety) : "Unavailable";
  confidence.textContent = Number.isFinite(workstation.confidence)
    ? `${workstation.confidence}%`
    : "--";
  peak.textContent = Number.isFinite(workstation.peak) ? `${workstation.peak.toFixed(1)}°C` : "--";
  change.textContent = timeAgo(workstation.lastChanged);
  safety.textContent = liveConnected ? titleCaseState(workstation.safety) : "--";

  const timeline = document.getElementById("workstation-timeline");
  timeline.innerHTML = "";
  workstation.history.forEach((sample, index) => {
    const segment = document.createElement("span");
    segment.className = sample === null ? "is-missing" : sample ? "is-occupied" : "is-free";
    segment.setAttribute("role", "listitem");
    segment.setAttribute(
      "aria-label",
      `Reading ${index + 1}: ${sample === null ? "unavailable" : sample ? "occupied" : "not occupied"}`,
    );
    timeline.appendChild(segment);
  });
}

function renderEventLog() {
  ids.eventLog.innerHTML = "";
  eventLog.forEach((event) => {
    const item = document.createElement("li");
    const message = document.createElement("strong");
    const time = document.createElement("span");
    message.textContent = event.message;
    time.textContent = formatTime(new Date(event.time));
    item.append(message, time);
    ids.eventLog.appendChild(item);
  });
  ids.eventCount.textContent = `${eventLog.length} events`;
}

function drawOfflineField() {
  const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, "#0d1227");
  gradient.addColorStop(0.5, "#11162b");
  gradient.addColorStop(1, "#09101f");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(255,255,255,0.035)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= canvas.width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= canvas.height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
}

function render() {
  ids.lastUpdated.textContent = liveConnected && Number.isFinite(lastSensorTimestamp)
    ? formatTime(new Date(lastSensorTimestamp))
    : "--";
  ids.syncLabel.textContent = liveConnected
    ? "Live sensor feed"
    : connectionState === "connecting"
      ? "Connecting"
      : connectionState === "stale"
        ? "Sensor feed stale"
        : "Sensor offline";
  ids.syncDot.classList.toggle("is-live", liveConnected);
  ids.syncDot.classList.toggle("is-offline", !liveConnected);
  ids.detectionMode.textContent = liveConnected ? workstation.detectionMode || "--" : "--";
  ids.connectionState.textContent = liveConnected
    ? "Live sensor feed"
    : connectionState === "connecting"
      ? "Connecting"
      : connectionState === "stale"
        ? "Stale feed"
        : "Offline";
  ids.stackSnapshotInterval.textContent = `Updated every ${snapshotIntervalSeconds} seconds`;

  if (!liveConnected) {
    ids.frameLabel.textContent = connectionState === "connecting" ? "Connecting" : "Offline";
    ids.snapshotStatus.textContent = connectionState === "connecting" ? "Waiting" : "Unavailable";
    ids.thermalCaption.textContent =
      "No current thermal frame is available. The dashboard will reconnect automatically.";
    ids.feedPlaceholderTitle.textContent =
      connectionState === "connecting" ? "Connecting to the sensor" : "Live feed unavailable";
    ids.feedPlaceholderText.textContent =
      connectionState === "stale"
        ? "The Raspberry Pi status file has stopped updating."
        : "Start the Raspberry Pi monitor to restore live data.";
  } else {
    ids.thermalCaption.textContent =
      `Thermal preview image exported from the Raspberry Pi monitor every ${snapshotIntervalSeconds} seconds.`;
  }

  renderWorkstation();
  renderEventLog();
}

ids.thermalSnapshot.addEventListener("load", () => {
  if (!liveConnected) return;
  ids.thermalSnapshot.classList.add("is-visible");
  ids.feedPlaceholder.hidden = true;
  ids.snapshotStatus.textContent = "Live image";
});

ids.thermalSnapshot.addEventListener("error", () => {
  ids.thermalSnapshot.classList.remove("is-visible");
  ids.feedPlaceholder.hidden = false;
  ids.snapshotStatus.textContent = liveConnected ? "Image pending" : "Unavailable";
});

drawOfflineField();
render();
pollLiveStatus();
setInterval(pollLiveStatus, 1000);
setInterval(render, 1000);
setInterval(() => refreshThermalSnapshot(true), 30000);
