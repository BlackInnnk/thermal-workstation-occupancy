# Thermal Workstation Occupancy

A thermal imaging based workstation occupancy detection prototype for welding benches.

## Project Structure

- `dashboard/` - Web dashboard prototype for displaying workstation occupancy status.
- `sensor/` - Raspberry Pi and Lepton thermal camera scripts for frame capture, ROI analysis, and occupancy detection.

## Current Status

- Web dashboard prototype supports one live workstation and falls back to simulated data when the Raspberry Pi status file is unavailable.
- Lepton thermal camera capture has been tested with `pylepton`.
- Sensor-side ROI viewing script is available under `sensor/`.

## Run Sensor ROI Viewer on Raspberry Pi

Install dependencies on the Raspberry Pi:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-spidev
```

Run the viewer:

```bash
sudo python3 sensor/thermal_roi_viewer.py
```

Adjust ROI boxes interactively:

```bash
sudo python3 sensor/adjust_rois.py
```

Controls:

```text
1 / 2 = select ROI
Arrow keys or WASD = move selected ROI
+ / - = scale selected ROI
J / L = shrink / grow width
K / I = shrink / grow height
Enter = save ROI positions to thermal_roi_viewer.py and exit
Q = quit without saving
```

Useful options:

```bash
sudo python3 sensor/thermal_roi_viewer.py --threshold 30 --scale 10
```

## Collect Labelled Thermal Data

Use this after the Lepton camera is installed in the real workstation position.

```bash
sudo python3 sensor/collect_thermal_dataset.py
```

Keyboard controls in the OpenCV preview window:

```text
Enter = start / stop recording
f = free
o = occupied
c = cooling after switch-off
h = hot_empty / unattended hot tool
q = quit
```

Data is saved under `data/raw/<session>/` and is ignored by git.

Label meanings:

| Label | Situation | Occupancy | Safety |
|---|---|---|---|
| `free` | No person, workstation is cold | Free | Normal |
| `occupied` | Person is seated or standing at the workstation | Occupied | Normal |
| `cooling` | No person, tool has been switched off but is still cooling | Free | Cooling |
| `hot_empty` | No person, tool is still on or dangerously hot | Free | Alert |

## Analyse Collected Dataset

After copying a dataset session from the Raspberry Pi to this computer, generate a quick summary and visual checks:

```bash
python3 -m pip install numpy pillow
```

```bash
python3 sensor/analyse_dataset.py data/raw/<session>
```

If no session is provided, the script analyses the latest folder under `data/raw/`.

```bash
python3 sensor/analyse_dataset.py
```

Outputs are written to `data/analysis/<session>/`:

```text
summary.txt
summary.json
label_counts.csv
temperature_summary.csv
label_distribution.png
label_timeline.png
max_temperature_timeline.png
sample_grid.png
```

Use this before training to check label balance, temperature ranges, and whether the saved examples match the intended class labels.

## Train Neural-Network Baselines

The training script uses a small NumPy MLP instead of Edge Impulse. This keeps the training process local, reproducible, and easy to explain in the dissertation. Edge Impulse can still be considered later for comparison or deployment experiments, after the data pipeline and labels are validated.

The recommended model for the live system is binary occupancy detection:

```bash
python3 sensor/train_thermal_mlp.py \
  data/raw/20260611_195455 \
  data/raw/free_neighbor_person_01 \
  data/raw/train_02 \
  --task occupancy \
  --run-name occupancy_mlp_train02_relabel
```

This model is trained on the full 80x60 radiometric thermal frame. It outputs two classes:

```text
not_occupied
occupied
```

The original labels are mapped as follows:

```text
occupied -> occupied
free, cooling, hot_empty -> not_occupied
```

This is the main deep-learning task because human occupancy is a spatial recognition problem. Residual heat states are handled by temporal ROI rules because `cooling`, `hot_empty`, and `free` can look similar in a single thermal frame.

For comparison, the same script can still train the four-class state baseline:

```bash
python3 sensor/train_thermal_mlp.py \
  data/raw/20260611_195455 \
  data/raw/free_neighbor_person_01 \
  data/raw/train_02 \
  --task state \
  --run-name mlp_train02_relabel
```

The four-class model outputs:

```text
free
occupied
cooling
hot_empty
```

Outputs are written to `models/<run>/` and ignored by git:

```text
model.npz
metrics.json
training_curves.csv
confusion_matrix.csv
confusion_matrix.png
```

Important limitation: this first script uses a random frame-level split, so nearby frames from the same recording can appear in both train and test sets. This is useful for a first proof of concept, but the final evaluation should also test on a separate recording session.

Evaluate a saved model on a separate recording session:

```bash
python3 sensor/evaluate_model.py \
  models/<run>/model.npz \
  data/raw/eval_01
```

Evaluation outputs are written to `models/<run>/evaluations/<session>/`:

```text
metrics.json
class_metrics.csv
predictions.csv
confusion_matrix.csv
confusion_matrix.png
```

## Run Hybrid Live Monitor

The monitor keeps human occupancy and thermal safety as separate state machines. With no model argument it uses ROI rules for human detection:

```bash
sudo python3 sensor/workstation_monitor.py
```

For the final hybrid system, pass the binary occupancy model:

```bash
sudo python3 sensor/workstation_monitor.py \
  --occupancy-model models/occupancy_mlp_train02_relabel/model.npz \
  --occupied-confirm 3 \
  --leave-confirm 10
```

In this mode:

```text
Deep learning: occupied / not_occupied
Rule-based temporal logic: safe / monitoring / cooling / unattended hot
```

Occupancy states:

| State | Meaning |
|---|---|
| `FREE` | No confirmed person and no recent use |
| `OCCUPIED` | A human-sized connected thermal region has persisted for 5 seconds |
| `RECENTLY_USED` | The person has left, but the workstation was used within the last 15 minutes |

Safety states:

| State | Meaning |
|---|---|
| `IN_USE` | The workstation is occupied |
| `SAFE` | Tool-area temperature has stayed below the safe threshold long enough to confirm it is safe |
| `MONITORING` | The workstation is empty and warm; more trend data is needed |
| `COOLING` | Tool-area temperature is decreasing, or has dropped from a recent hot peak but is not yet confirmed safe |
| `UNATTENDED_HOT` | The workstation is empty and remains above the alert threshold without sufficient cooling |

Default baseline parameters:

```text
Human threshold: max(27 C, ambient + 4 C)
Human connected component: at least 2.5% of Human Area, minimum 20 pixels
Occupied confirmation: 5 seconds
Leave confirmation: 15 seconds
Recently used duration: 15 minutes
Safe tool temperature: below 38 C
Hot tool alert threshold: 45 C
Cooling trend: at least -0.5 C/min
Cooling drop from recent hot peak: at least 2 C
Safe confirmation: below 38 C for 60 seconds
Unattended hot delay: 3 minutes
```

The safety state uses hysteresis: after a tool has been visibly hot, the system can remain in `COOLING` even when the measured temperature has just fallen below 38 C. It only becomes `SAFE` after the tool-area temperature has stayed below the safe threshold for the confirmation period.

These are initial engineering values, not final research results. They should be calibrated from the labelled dataset.

The monitor also writes machine-readable status and a dashboard thermal preview to:

```text
data/runtime/status.json
data/runtime/thermal_view.jpg
```

## Run Live Dashboard

The easiest way to run the full Raspberry Pi system is:

```bash
cd ~/thermal-workstation-occupancy
./scripts/start_system.sh
```

This starts both parts in the background:

```text
sensor/workstation_monitor.py
python3 -m http.server 8000
```

Then open the dashboard:

```text
http://<raspberry-pi-ip>:8000/dashboard/
```

If using Tailscale:

```text
http://<tailscale-ip>:8000/dashboard/
```

Replace `<tailscale-ip>` with the Raspberry Pi's Tailscale address.

To stop the system:

```bash
cd ~/thermal-workstation-occupancy
./scripts/stop_system.sh
```

Logs are written to:

```text
data/runtime/logs/monitor.log
data/runtime/logs/dashboard.log
```

To watch logs while the system is running:

```bash
tail -f data/runtime/logs/monitor.log
```

If you want to run the two parts manually instead, run the monitor first so `data/runtime/status.json` is updated:

```bash
sudo python3 sensor/workstation_monitor.py \
  --occupancy-model models/occupancy_mlp_train02_relabel/model.npz \
  --occupied-confirm 3 \
  --leave-confirm 10 \
  --snapshot-interval 30 \
  --no-window
```

In a second terminal, serve the project root:

```bash
cd ~/thermal-workstation-occupancy
python3 -m http.server 8000
```

When `data/runtime/status.json` is available, the dashboard shows the live Raspberry Pi monitor state for the workstation. The thermal preview image is refreshed from `data/runtime/thermal_view.jpg` every 30 seconds. If the status file is unavailable, it falls back to the built-in demo simulation.

Useful tuning example:

```bash
sudo python3 sensor/workstation_monitor.py \
  --human-delta 4 \
  --occupied-confirm 5 \
  --leave-confirm 15 \
  --tool-safe 38 \
  --tool-alert 45 \
  --unattended-delay-seconds 180
```

Run the state-logic tests:

```bash
python3 -m unittest discover -s tests -v
```

## Planned Pipeline

```text
FLIR Lepton -> Raspberry Pi -> Python/OpenCV ROI analysis -> Occupancy status -> Web dashboard
```
