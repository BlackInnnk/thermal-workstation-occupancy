const benches = [
  {
    id: 1,
    name: "Welding Bench 1",
    occupied: true,
    confidence: 88,
    peak: 34.8,
    lastChanged: Date.now() - 1000 * 70,
    history: [false, false, true, true, true, true, true, true, true, true, true, true],
  },
  {
    id: 2,
    name: "Welding Bench 2",
    occupied: false,
    confidence: 93,
    peak: 27.4,
    lastChanged: Date.now() - 1000 * 260,
    history: [false, false, false, true, true, false, false, false, false, false, false, false],
  },
];

let autoUpdate = true;
let frameCount = 0;
let residualModeUntil = 0;
let eventLog = [
  { time: Date.now() - 1000 * 70, message: "Bench 1 changed to Occupied" },
  { time: Date.now() - 1000 * 260, message: "Bench 2 changed to Free" },
];

const canvas = document.getElementById("heatmapCanvas");
const ctx = canvas.getContext("2d");

const ids = {
  lastUpdated: document.getElementById("lastUpdated"),
  occupiedCount: document.getElementById("occupiedCount"),
  freeCount: document.getElementById("freeCount"),
  frameLabel: document.getElementById("frameLabel"),
  eventCount: document.getElementById("eventCount"),
  eventLog: document.getElementById("eventLog"),
  simulationState: document.getElementById("simulationState"),
  toggleAuto: document.getElementById("toggleAuto"),
  toggleBench1: document.getElementById("toggleBench1"),
  toggleBench2: document.getElementById("toggleBench2"),
  residualTest: document.getElementById("residualTest"),
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

function setBenchStatus(index, occupied, source = "Manual") {
  const bench = benches[index];
  if (bench.occupied !== occupied) {
    bench.occupied = occupied;
    bench.lastChanged = Date.now();
    bench.history.push(occupied);
    bench.history = bench.history.slice(-12);
    addEvent(`${source}: ${bench.name} changed to ${occupied ? "Occupied" : "Free"}`);
  }
  bench.confidence = occupied ? randomInt(82, 96) : randomInt(88, 98);
  bench.peak = occupied ? randomFloat(33.2, 36.8) : randomFloat(25.4, 28.7);
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomFloat(min, max) {
  return Math.round((Math.random() * (max - min) + min) * 10) / 10;
}

function maybeAutoUpdate() {
  if (!autoUpdate) return;

  benches.forEach((bench, index) => {
    if (Math.random() < 0.16) {
      setBenchStatus(index, !bench.occupied, "Auto");
    } else {
      bench.history.push(bench.occupied);
      bench.history = bench.history.slice(-12);
      bench.confidence = bench.occupied ? randomInt(84, 96) : randomInt(89, 98);
      bench.peak = bench.occupied ? randomFloat(33.0, 36.9) : randomFloat(25.2, 28.9);
    }
  });
}

function renderBench(bench) {
  const panel = document.getElementById(`bench-${bench.id}-panel`);
  const dot = document.getElementById(`bench-${bench.id}-dot`);
  const status = document.getElementById(`bench-${bench.id}-status`);
  const confidence = document.getElementById(`bench-${bench.id}-confidence`);
  const peak = document.getElementById(`bench-${bench.id}-peak`);
  const change = document.getElementById(`bench-${bench.id}-change`);

  panel.classList.toggle("is-occupied", bench.occupied);
  panel.classList.toggle("is-free", !bench.occupied);
  dot.classList.toggle("is-occupied", bench.occupied);
  dot.classList.toggle("is-free", !bench.occupied);

  status.textContent = bench.occupied ? "Occupied" : "Free";
  confidence.textContent = `${bench.confidence}%`;
  peak.textContent = `${bench.peak.toFixed(1)}C`;
  change.textContent = timeAgo(bench.lastChanged);

  const timeline = document.getElementById(`bench-${bench.id}-timeline`);
  timeline.innerHTML = "";
  bench.history.forEach((sample) => {
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

function drawHeatmap() {
  const lowWidth = 80;
  const lowHeight = 60;
  const cellW = canvas.width / lowWidth;
  const cellH = canvas.height / lowHeight;
  const now = Date.now();

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < lowHeight; y += 1) {
    for (let x = 0; x < lowWidth; x += 1) {
      let value = 0.18 + Math.sin((x + frameCount) * 0.08) * 0.025 + Math.cos(y * 0.18) * 0.02;

      if (benches[0].occupied) {
        value += gaussian(x, y, 25 + Math.sin(frameCount * 0.07) * 2, 31, 7, 10, 0.72);
      }
      if (benches[1].occupied) {
        value += gaussian(x, y, 57 + Math.cos(frameCount * 0.06) * 2, 30, 7, 10, 0.72);
      }

      if (now < residualModeUntil) {
        value += gaussian(x, y, 58, 40, 2, 2, 0.8);
      }

      value += Math.random() * 0.025;
      value = Math.max(0, Math.min(1, value));

      ctx.fillStyle = thermalColor(value);
      ctx.fillRect(Math.floor(x * cellW), Math.floor(y * cellH), Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

function render() {
  const occupied = benches.filter((bench) => bench.occupied).length;
  ids.occupiedCount.textContent = occupied;
  ids.freeCount.textContent = benches.length - occupied;
  ids.lastUpdated.textContent = formatTime(new Date());
  ids.frameLabel.textContent = `Frame ${String(frameCount).padStart(4, "0")}`;
  ids.simulationState.textContent = autoUpdate ? "Auto update on" : "Auto update paused";

  benches.forEach(renderBench);
  renderEventLog();
  drawHeatmap();
}

ids.toggleAuto.addEventListener("click", () => {
  autoUpdate = !autoUpdate;
  ids.toggleAuto.textContent = autoUpdate ? "Pause auto" : "Resume auto";
  addEvent(`Simulation ${autoUpdate ? "resumed" : "paused"}`);
  render();
});

ids.toggleBench1.addEventListener("click", () => {
  setBenchStatus(0, !benches[0].occupied);
  render();
});

ids.toggleBench2.addEventListener("click", () => {
  setBenchStatus(1, !benches[1].occupied);
  render();
});

ids.residualTest.addEventListener("click", () => {
  setBenchStatus(1, false, "Residual filter");
  residualModeUntil = Date.now() + 7000;
  addEvent("Residual heat detected in Bench 2 ROI; workstation remains Free");
  render();
});

setInterval(() => {
  frameCount += 1;
  maybeAutoUpdate();
  render();
}, 1800);

render();
