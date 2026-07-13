const workstation = {
  id: 1,
  name: "Welding Workstation",
  occupied: false,
  displayStatus: "FREE",
  confidence: 0,
  peak: 27.4,
  safety: "SAFE",
  lastChanged: Date.now(),
  history: [false, false, false, false, false, false, false, false, false, false, false, false],
};

let autoUpdate = true;
let liveConnected = false;
let lastLiveStatusAt = 0;
let frameCount = 0;
let lastSnapshotRefreshAt = 0;
let thermalSnapshotUrl = "/data/runtime/thermal_view.jpg";
let eventLog = [{ time: Date.now(), message: "Workstation initialised as Free" }];

const canvas = document.getElementById("heatmapCanvas");
const ctx = canvas.getContext("2d");

const ids = {
  lastUpdated: document.getElementById("lastUpdated"),
  frameLabel: document.getElementById("frameLabel"),
  eventCount: document.getElementById("eventCount"),
  eventLog: document.getElementById("eventLog"),
  syncLabel: document.getElementById("syncLabel"),
  syncDot: document.getElementById("syncDot"),
  detectionMode: document.getElementById("detectionMode"),
  snapshotStatus: document.getElementById("snapshotStatus"),
  thermalCaption: document.getElementById("thermalCaption"),
  simulationState: document.getElementById("simulationState"),
  thermalSnapshot: document.getElementById("thermalSnapshot"),
};

function formatTime(date) {
  return new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function timeAgo(timestamp) {
  const seconds = Math.max(1, Math.round((Date.now() - timestamp) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  return `${minutes}m ago`;
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
  return "is-safe";
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

function setWorkstationStatus(occupied, source = "Manual") {
  const nextStatus = occupied ? "OCCUPIED" : "FREE";
  if (workstation.occupied !== occupied || workstation.displayStatus !== nextStatus) {
    workstation.occupied = occupied;
    workstation.displayStatus = nextStatus;
    workstation.lastChanged = Date.now();
    workstation.history.push(occupied);
    workstation.history = workstation.history.slice(-12);
    addEvent(`${source}: ${workstation.name} changed to ${occupied ? "Occupied" : "Free"}`);
  }

  workstation.confidence = occupied ? randomInt(82, 96) : randomInt(88, 98);
  workstation.peak = occupied ? randomFloat(33.2, 36.8) : randomFloat(25.4, 28.7);
  workstation.safety = occupied ? "IN_USE" : "SAFE";
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomFloat(min, max) {
  return Math.round((Math.random() * (max - min) + min) * 10) / 10;
}

function maybeAutoUpdate() {
  if (!autoUpdate || liveConnected) return;

  if (Math.random() < 0.16) {
    setWorkstationStatus(!workstation.occupied, "Auto");
  } else {
    workstation.history.push(workstation.occupied);
    workstation.history = workstation.history.slice(-12);
    workstation.confidence = workstation.occupied ? randomInt(84, 96) : randomInt(89, 98);
    workstation.peak = workstation.occupied ? randomFloat(33.0, 36.9) : randomFloat(25.2, 28.9);
    workstation.safety = workstation.occupied ? "IN_USE" : "SAFE";
  }
}

function applyLiveStatus(payload) {
  const occupancyState = payload?.occupancy?.state || "FREE";
  const safetyState = payload?.safety?.state || "SAFE";
  const occupied = occupancyState === "OCCUPIED";
  const modelProbability = payload?.model?.occupied_probability;
  const toolTemperature = payload?.safety?.tool_temperature_c ?? payload?.metrics?.tool_p95_c;
  thermalSnapshotUrl = normaliseRuntimeUrl(payload?.snapshot?.url, thermalSnapshotUrl);
  const changedAt = liveStateChangedAt(payload);

  liveConnected = true;
  lastLiveStatusAt = Date.now();

  if (
    workstation.occupied !== occupied ||
    workstation.safety !== safetyState ||
    workstation.displayStatus !== occupancyState
  ) {
    addEvent(
      `Live: ${workstation.name} ${titleCaseState(occupancyState)}, safety ${titleCaseState(safetyState)}`,
    );
    workstation.lastChanged = changedAt ?? Date.now();
  }

  workstation.occupied = occupied;
  workstation.displayStatus = occupancyState;
  workstation.confidence = Number.isFinite(modelProbability)
    ? Math.round(modelProbability * 100)
    : occupied
      ? 100
      : 0;
  workstation.peak = Number.isFinite(toolTemperature) ? Number(toolTemperature) : workstation.peak;
  workstation.safety = safetyState;
  if (changedAt !== null) {
    workstation.lastChanged = changedAt;
  }
  workstation.history.push(occupied);
  workstation.history = workstation.history.slice(-12);
}

async function pollLiveStatus() {
  try {
    const response = await fetch(`/data/runtime/status.json?ts=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    applyLiveStatus(payload);
  } catch {
    if (liveConnected && Date.now() - lastLiveStatusAt > 5000) {
      liveConnected = false;
      ids.thermalSnapshot.classList.remove("is-visible");
      addEvent("Live status disconnected; dashboard returned to demo mode");
    }
  }
}

function refreshThermalSnapshot(force = false) {
  if (!liveConnected) return;
  const now = Date.now();
  if (!force && now - lastSnapshotRefreshAt < 30000) return;

  lastSnapshotRefreshAt = now;
  ids.thermalSnapshot.src = `${thermalSnapshotUrl}?ts=${now}`;
  ids.frameLabel.textContent = `Snapshot ${formatTime(new Date(now))}`;
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
  panel.classList.toggle("is-warning", !workstation.occupied && isWarningSafety(workstation.safety));
  panel.classList.toggle("is-danger", workstation.safety === "UNATTENDED_HOT");
  dot.classList.toggle("is-occupied", workstation.occupied);
  dot.classList.toggle("is-free", workstation.displayStatus === "FREE");
  dot.classList.toggle("is-recent", workstation.displayStatus === "RECENTLY_USED");
  dot.classList.toggle("is-warning", workstation.displayStatus === "RECENTLY_USED");

  safetyCard.className = `state-card state-card-safety ${safetyStateClass}`;
  safetyDot.className = `status-indicator safety-indicator ${safetyStateClass}`;

  status.textContent = titleCaseState(workstation.displayStatus);
  safetyPrimary.textContent = titleCaseState(workstation.safety);
  confidence.textContent = `${workstation.confidence}%`;
  peak.textContent = `${workstation.peak.toFixed(1)}°C`;
  change.textContent = timeAgo(workstation.lastChanged);
  safety.textContent = titleCaseState(workstation.safety);

  const timeline = document.getElementById("workstation-timeline");
  timeline.innerHTML = "";
  workstation.history.forEach((sample) => {
    const segment = document.createElement("span");
    segment.className = sample ? "is-occupied" : "is-free";
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

function thermalColor(value) {
  const stops = [
    [0.0, [28, 42, 104]],
    [0.32, [22, 117, 173]],
    [0.55, [66, 184, 131]],
    [0.76, [242, 193, 78]],
    [1.0, [255, 95, 87]],
  ];

  for (let i = 0; i < stops.length - 1; i += 1) {
    const [start, from] = stops[i];
    const [end, to] = stops[i + 1];
    if (value <= end) {
      const t = (value - start) / (end - start);
      const rgb = from.map((channel, idx) => Math.round(channel + (to[idx] - channel) * t));
      return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    }
  }
  return "rgb(255, 95, 87)";
}

function gaussian(x, y, cx, cy, sx, sy, strength) {
  const dx = (x - cx) / sx;
  const dy = (y - cy) / sy;
  return Math.exp(-(dx * dx + dy * dy) / 2) * strength;
}

function drawFallbackHeatmap() {
  if (liveConnected && ids.thermalSnapshot.classList.contains("is-visible")) return;

  const lowWidth = 80;
  const lowHeight = 60;
  const cellW = canvas.width / lowWidth;
  const cellH = canvas.height / lowHeight;
  const now = Date.now();

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < lowHeight; y += 1) {
    for (let x = 0; x < lowWidth; x += 1) {
      let value = 0.18 + Math.sin((x + frameCount) * 0.08) * 0.025 + Math.cos(y * 0.18) * 0.02;

      if (workstation.occupied) {
        value += gaussian(x, y, 55 + Math.sin(frameCount * 0.07) * 2, 30, 7, 10, 0.72);
      }
      if (!workstation.occupied && isWarningSafety(workstation.safety)) {
        value += gaussian(x, y, 18, 40, 2, 2, 0.8);
      }

      value += Math.random() * 0.025;
      value = Math.max(0, Math.min(1, value));

      ctx.fillStyle = thermalColor(value);
      ctx.fillRect(Math.floor(x * cellW), Math.floor(y * cellH), Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

function render() {
  ids.lastUpdated.textContent = formatTime(new Date());
  if (!liveConnected) ids.frameLabel.textContent = `Demo frame ${String(frameCount).padStart(4, "0")}`;
  ids.syncLabel.textContent = liveConnected ? "Live sensor feed" : "Demo fallback";
  ids.syncDot.classList.toggle("is-live", liveConnected);
  ids.syncDot.classList.toggle("is-offline", !liveConnected);
  ids.detectionMode.textContent = liveConnected ? "ML + rules" : "Demo fallback";
  if (!liveConnected) ids.snapshotStatus.textContent = "Demo image";
  ids.thermalCaption.textContent = liveConnected
    ? "Thermal preview image exported from the Raspberry Pi monitor every 30 seconds."
    : "Simulated 80x60 radiometric frame for dashboard demonstration.";
  ids.simulationState.textContent = liveConnected
    ? "Live sensor feed"
    : autoUpdate
      ? "Demo fallback"
      : "Demo paused";

  renderWorkstation();
  renderEventLog();
  refreshThermalSnapshot();
  drawFallbackHeatmap();
}

ids.thermalSnapshot.addEventListener("load", () => {
  ids.thermalSnapshot.classList.add("is-visible");
  ids.snapshotStatus.textContent = "Live image";
});

ids.thermalSnapshot.addEventListener("error", () => {
  ids.thermalSnapshot.classList.remove("is-visible");
  ids.snapshotStatus.textContent = liveConnected ? "Image pending" : "Demo image";
});

setInterval(() => {
  frameCount += 1;
  maybeAutoUpdate();
  render();
}, 1800);

setInterval(pollLiveStatus, 1000);
setInterval(() => refreshThermalSnapshot(true), 30000);
pollLiveStatus();
render();
