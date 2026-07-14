# Hot Seat

Hot Seat is a privacy-friendly thermal monitoring system for a shared soldering workstation. A ceiling-mounted FLIR Lepton 2.5 Radiometric sensor and Raspberry Pi classify human occupancy locally, track residual tool heat over time, and publish a live web dashboard.

The deployed lab system monitors one workstation in UCL One Pool Street, Lab 107. A separate simplified device is used for the interactive exhibition demonstration.

## What the system reports

Occupancy and tool safety are deliberately separated:

| Output | States |
|---|---|
| Occupancy | `FREE`, `OCCUPIED`, `RECENTLY_USED` |
| Safety | `SAFE`, `IN_USE`, `MONITORING`, `COOLING`, `UNATTENDED_HOT` |

Human presence is inferred by a small binary neural network. Tool safety is evaluated with temperature thresholds, cooling trends, delays, and hysteresis. This hybrid design is used because presence is a spatial image-recognition problem, while cooling and unattended heat depend on change over time and cannot be determined reliably from one frame.

## System architecture

```text
FLIR Lepton 2.5 Radiometric (80x60)
              |
          SPI + I2C
              |
         Raspberry Pi 4
              |
   +----------+-----------+
   |                      |
Binary occupancy MLP   Tool ROI rules
   |                      |
   +----------+-----------+
              |
     temporal state machines
              |
 status.json + thermal_view.jpg
              |
       dashboard server
              |
  local network / Tailscale Funnel
```

All inference runs on the Raspberry Pi. The raw `.npy` dataset is neither served by the dashboard nor committed to Git.

## Repository layout

```text
index.html                         project landing page
assets/                            deployed-device and evidence images
dashboard/                         live dashboard HTML, CSS, and JavaScript
sensor/                            capture, labelling, analysis, training, and monitoring
scripts/                           start, stop, status, server, and public-tunnel scripts
tests/                             state-machine, model-artifact, and server tests
models/occupancy_mlp_train02_relabel/
                                   final deployment model and compact reports
requirements-analysis.txt          Mac/analysis dependencies
```

`data/`, experimental model runs, local IP notes, and QR images are intentionally ignored.

## Web routes

The custom Python server exposes only the project UI and current runtime outputs:

| Route | Purpose |
|---|---|
| `/` or `/dashboard/` | Project landing page |
| `/dashboard/live/` | Live workstation dashboard |
| `/healthz` | Server and sensor freshness as JSON |
| `/data/runtime/status.json` | Current machine-readable state |
| `/data/runtime/events.json` | Recent occupancy and safety state changes |
| `/data/runtime/thermal_view.jpg` | Low-resolution thermal preview |

The server does not expose `data/raw/`, training data, source code, logs, or model files.

## Lab device quick start

Run these commands on the installed Raspberry Pi from the repository root:

```bash
cd ~/thermal-workstation-occupancy
./scripts/start_system.sh
```

This starts the Lepton monitor and dashboard server, then waits until both the HTTP service and sensor feed are healthy. Open:

```text
Project page:   http://<raspberry-pi-ip>:8000/dashboard/
Live dashboard: http://<raspberry-pi-ip>:8000/dashboard/live/
```

After a cold start or a sensor interruption, safety begins in `MONITORING` and requires 60 seconds of valid below-threshold observations before reporting `SAFE`.

Check the complete system at any time:

```bash
./scripts/status_system.sh
curl -s http://127.0.0.1:8000/healthz
```

A healthy response contains:

```json
{"service":"hot-seat-dashboard","server":"ok","sensor":"fresh","status_age_seconds":0.5}
```

Stop the local system:

```bash
./scripts/stop_system.sh
```

Logs are written to:

```text
data/runtime/logs/monitor.log
data/runtime/logs/dashboard.log
```

Follow the sensor log:

```bash
tail -f data/runtime/logs/monitor.log
```

The startup script uses the tracked model at:

```text
models/occupancy_mlp_train02_relabel/model.npz
```

To select another model or port without editing the script:

```bash
OCCUPANCY_MODEL=models/<run>/model.npz DASHBOARD_PORT=8080 ./scripts/start_system.sh
```

## Public dashboard

The submitted public exhibition QR code points to the stable Tailscale Funnel project route (`/dashboard/`). It intentionally opens the landing page first; visitors can then open the live monitor at `/dashboard/live/`. To start the monitor, dashboard server, and Funnel together:

```bash
cd ~/thermal-workstation-occupancy
./scripts/start_public_system.sh
```

The Funnel root URL should open the project page. Add `/dashboard/live/` for the live monitor.

Stop the public tunnel and local services:

```bash
./scripts/stop_public_system.sh
```

The public URL works only while the Raspberry Pi is powered, online, signed in to Tailscale, and the public startup script is running. The dashboard now shows an explicit stale or offline state when sensor updates stop; it does not substitute random demo data.

Public startup checks Tailscale before starting local services. If Funnel startup fails, the script rolls back only the local processes that it started during that attempt and retains PID files for any process that could not be stopped safely.

Tailscale Funnel is intentionally public and unauthenticated. Disable it when a public feed is not required.

## Hardware setup

Tested lab hardware:

- Raspberry Pi 4
- FLIR Lepton 2.5 Radiometric, 80x60
- SparkFun/GroupGets-style Lepton breakout board
- Active Raspberry Pi cooling

### Lepton breakout wiring

The following physical Raspberry Pi header pins match the tested CE0 configuration:

| Breakout | Raspberry Pi signal | Physical pin |
|---|---|---|
| `3-5V` | 3.3 V | 1 |
| `GND` | Ground | 6 |
| `SDA` | GPIO 2 / SDA1 | 3 |
| `SCL` | GPIO 3 / SCL1 | 5 |
| `CLK` | GPIO 11 / SCLK | 23 |
| `MISO` | GPIO 9 / MISO | 21 |
| `MOSI` | GPIO 10 / MOSI | 19 |
| `CS` | GPIO 8 / CE0 | 24 |

Use the breakout's `3-5V` input as documented for that board. Do not connect multiple power inputs simultaneously, and switch off the Pi before changing wiring.

Enable I2C and SPI with `sudo raspi-config`, then reboot. Verify the interfaces:

```bash
i2cdetect -y 1
ls -l /dev/spidev0.0
cat /sys/module/spidev/parameters/bufsiz
```

Expected results:

```text
I2C address: 0x2a
SPI device:  /dev/spidev0.0
SPI bufsiz:  at least 65535
```

If necessary, add this kernel parameter and reboot:

```text
spidev.bufsiz=65535
```

On current Raspberry Pi OS images, append it to the existing single line in
`/boot/firmware/cmdline.txt`; do not create a second line.

## Raspberry Pi software prerequisites

Install system packages:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-spidev python3-pip i2c-tools curl git
```

The monitor also requires the `pylepton` package to be importable by the system Python used through `sudo`. The tested source is [GroupGets pylepton](https://github.com/groupgets/pylepton). If it is not already installed, install the downloaded source into the Pi system Python:

```bash
sudo python3 -m pip install --break-system-packages /path/to/pylepton
```

Verify all runtime imports before deployment:

```bash
sudo python3 - <<'PY'
import cv2
import numpy
from pylepton import Lepton
print("Runtime imports OK")
PY
```

## ROI calibration and sensor checks

Run the thermal viewer from an active Raspberry Pi desktop or VNC terminal:

```bash
cd ~/thermal-workstation-occupancy
sudo --preserve-env=DISPLAY,XAUTHORITY python3 sensor/thermal_roi_viewer.py
```

Adjust both ROI boxes interactively:

```bash
sudo --preserve-env=DISPLAY,XAUTHORITY python3 sensor/adjust_rois.py
```

Controls:

```text
1 / 2             select ROI
Arrow keys / WASD move selected ROI
+ / -             resize both dimensions
J / L             shrink / grow width
K / I             shrink / grow height
Enter             save to thermal_roi_viewer.py and exit
Q                 quit without saving
```

The live monitor imports the same `DEFAULT_ROIS`, so saved calibration applies automatically after the monitor is restarted. The full-frame occupancy model does not depend on the Human Area ROI, but the Tool Area ROI remains critical for safety classification.

## Live detection logic

The deployment startup script uses these values:

```text
Occupied confirmation:       3 seconds
Leave confirmation:          10 seconds
Recently used duration:      15 minutes
Tool safe threshold:         below 38 C
Tool alert threshold:        45 C
Cooling trend:               -0.5 C/min or faster
Minimum drop from hot peak:  2 C
Trend observation minimum:   45 seconds
Unattended alert delay:      3 minutes
Safe confirmation:           below 38 C for 60 seconds
Thermal web snapshot:        every 30 seconds
Live status JSON:            every 1 second, plus immediate state changes
State change log:            recent occupancy and safety transitions in data/runtime/events.json
Sensor gap reset:            after more than 2 seconds without a valid frame
```

Occupancy transitions:

```text
FREE -> OCCUPIED -> RECENTLY_USED -> FREE
```

`RECENTLY_USED` records workstation history for 15 minutes after confirmed departure. Safety is calculated independently, so the dashboard can show `RECENTLY_USED + COOLING`, `RECENTLY_USED + UNATTENDED_HOT`, or `RECENTLY_USED + SAFE`.

Safety interpretation:

| State | Meaning |
|---|---|
| `IN_USE` | Occupancy is confirmed; an attended tool is not treated as an alert |
| `SAFE` | Tool ROI has remained below the safe threshold for the confirmation period |
| `MONITORING` | The bench is empty but more trend evidence is needed |
| `COOLING` | Temperature is falling or has dropped from a recent hot peak |
| `UNATTENDED_HOT` | The empty workstation remains hot without sufficient cooling evidence |

These values are engineering thresholds for this installation, not universal soldering-safety limits.

Long sensor gaps are not counted as continuous evidence. On recovery, pending enter/leave confirmation, cooling trends, unattended timers, and safe confirmation are restarted from fresh observations.

The monitor keeps inference at the sensor frame rate but limits `status.json` writes to once per second and routine log output to once per 10 seconds. State changes are written and logged immediately. This reduces unnecessary SD-card writes without slowing the dashboard.

## Dataset collection

OpenCV collection tools require a graphical desktop. Start a named session:

```bash
sudo --preserve-env=DISPLAY,XAUTHORITY python3 sensor/collect_thermal_dataset.py \
  --session <session-name> \
  --label free
```

Session names must be unique single folder names. If a previous failed attempt left an empty session, remove that folder or choose a new `--session` name; the collector now fails before creating a session when no graphical display is available.

Collector controls:

```text
Enter start or pause recording
F     free
O     occupied
C     cooling after switch-off
H     hot_empty / unattended hot condition
Q     save and quit
```

Label definitions:

| Label | Person | Tool condition |
|---|---|---|
| `free` | absent | cold/safe |
| `occupied` | present | on or off |
| `cooling` | absent | switched off and cooling |
| `hot_empty` | absent | on or persistently hot |

Frames are stored as radiometric `.npy` arrays because JPEG/PNG colour images discard the original per-pixel temperature values. Preview PNGs are saved only for visual inspection.

When collection was run with `sudo`, restore ownership before copying or analysing it:

```bash
sudo chown -R "$USER":"$(id -gn)" data/raw/<session-name>
```

## Dataset analysis and training

On the Mac or analysis computer, use an isolated Python environment. If a Conda
environment is already active, skip the first two commands and use that
environment's `python`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

Analyse a session:

```bash
python sensor/analyse_dataset.py data/raw/<session>
```

Outputs under `data/analysis/<session>/` include class counts, temperature summaries, timelines, and visual sample grids.

Train the binary occupancy model:

```bash
python sensor/train_thermal_mlp.py \
  data/raw/20260611_195455 \
  data/raw/free_neighbor_person_01 \
  data/raw/train_02 \
  --task occupancy \
  --run-name occupancy_mlp_retrained
```

Use a new run name for every experiment. The trainer deliberately refuses to
overwrite an existing model directory.

Training maps the four collection labels to two occupancy targets:

```text
occupied                 -> occupied
free, cooling, hot_empty -> not_occupied
```

Evaluate on an independently recorded session:

```bash
python sensor/evaluate_model.py \
  models/occupancy_mlp_train02_relabel/model.npz \
  data/raw/eval_01
```

The committed final model is a one-hidden-layer NumPy MLP with 64 hidden units and a 4,800-value input (80x60 Celsius frame). It contains its own training-set mean and standard deviation for inference.

The compact reports record the model SHA-256 and a SHA-256 fingerprint covering each source `labels.csv` plus every referenced `.npy` frame. These fingerprints allow local data copies to be checked against the reported experiment without publishing the privacy-sensitive raw dataset.

### Recorded evaluation

The final report contains:

```text
Training frames:              5,400
Independent evaluation:      1,592 frames
Independent session accuracy: 100% in the recorded Lab 107 session
```

The 100% result must not be interpreted as universal accuracy. Data was captured from the same fixed installation and adjacent video frames are strongly correlated. The independent session was excluded from training, but it still shares the same camera, room, workstation, and limited set of conditions. A stronger future evaluation should use additional people, clothing, seasons, tool positions, mounting shifts, and separate days.

The raw sessions are intentionally not included in the public repository, so a clean public clone can verify the model artifact and report fingerprints but cannot retrain or independently reproduce the recorded accuracy without the private dataset.

## Tests

Run the automated logic and health tests without camera hardware:

```bash
python -m unittest discover -s tests -v
```

Additional static checks:

```bash
python -m compileall -q sensor scripts tests
node --check dashboard/app.js
bash -n scripts/*.sh
```

Node.js is needed only for the JavaScript syntax check, not for the deployed dashboard.

### Raspberry Pi acceptance check

The automated tests do not exercise the physical Lepton camera, the deployed ROI,
or the public Tailscale route. After pulling a release onto the Raspberry Pi, run
this short acceptance check before an exhibition or unattended deployment:

```bash
cd ~/thermal-workstation-occupancy
git pull
./scripts/start_public_system.sh
./scripts/status_system.sh
curl -fsS http://127.0.0.1:8000/healthz
```

Confirm that `/healthz` reports `"sensor":"fresh"`, then check the live page on
both the local network and a phone using mobile data. Exercise at least these
transitions with the real bench and current camera position:

- `FREE` to `OCCUPIED` and back through `RECENTLY_USED`.
- `SAFE` or `MONITORING` to `IN_USE` while somebody is present.
- `COOLING` after the person leaves and the measured tool temperature falls.
- `UNATTENDED_HOT` when a stable hot tool remains above the alert conditions.

Finally, scan the submitted exhibition QR code and verify that it opens the
project page and can navigate to the live dashboard. ROI and temperature checks
must be repeated whenever the camera or soldering-iron holder is moved.

## Privacy and data handling

- No RGB camera or microphone is used.
- No face recognition, identity model, or person tracking is performed.
- Inference runs locally on the Raspberry Pi.
- Raw radiometric datasets remain outside Git and outside the web server.
- The public dashboard deliberately exposes current states and one low-resolution thermal preview approximately every 30 seconds.
- A public Tailscale Funnel URL is accessible to anyone who has the URL while the tunnel is active.

The thermal preview can show a coarse body silhouette. It is more privacy-preserving than RGB video, but it should not be described as containing no visual information.

## Known limitations

- The trained model is specific to one fixed 80x60 viewpoint and has limited participant diversity.
- Moving the camera requires new evaluation and may require retraining.
- Tool safety depends on the calibrated Tool Area ROI. A moved soldering iron or an insulating holder can reduce the measured temperature.
- Nearby people, reflective surfaces, heaters, sunlight, and overlapping warm equipment can affect readings.
- The system estimates occupancy, not identity or exact person count.
- Temporal state history resets when the monitor process restarts.
- The browser's event log and 12-reading strip are session-local displays, not a persistent usage database.
- Public availability depends on Raspberry Pi power, network connectivity, and Tailscale Funnel.
- This research prototype is not a certified fire, electrical, or occupational safety system.

## Troubleshooting

### Dashboard opens but says stale or offline

```bash
./scripts/status_system.sh
tail -50 data/runtime/logs/monitor.log
curl -s http://127.0.0.1:8000/healthz
```

`server: ok` with `sensor: stale` means the web server is running but `status.json` has stopped updating. Restart the local system after fixing the sensor error.

### Public QR code no longer responds

Confirm the Pi is online, then run:

```bash
./scripts/start_public_system.sh
./scripts/status_system.sh
```

The QR URL itself does not start the Pi or Funnel.

### I2C address `0x2a` is missing

Power down and recheck `3-5V`, ground, SDA, and SCL. Verify I2C is enabled before testing again.

### SPI reads only zeros or capture blocks

Power down and recheck CLK, MISO, MOSI, CS/CE0, and ground. Confirm `/dev/spidev0.0` exists and the SPI buffer size is at least 65535.

### OpenCV reports `could not connect to display`

Run the interactive script from a terminal inside the Raspberry Pi desktop/VNC session and preserve `DISPLAY` and `XAUTHORITY`. The background monitor uses `--no-window` and does not require VNC.

### Runtime directory permission error

The startup script attempts to repair this automatically. Manual recovery:

```bash
sudo install -d -m 775 -o "$(id -un)" -g "$(id -gn)" \
  data/runtime data/runtime/pids data/runtime/logs
```

## Exhibition device

The exhibition unit demonstrates the same sensing principle and state feedback with a second Raspberry Pi, Lepton sensor, display, and safe portable warm object. It is intentionally simplified for fast visitor interaction. The installed Lab 107 device remains the evaluated system and the source of the public live dashboard.
