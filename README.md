# Thermal Workstation Occupancy

A thermal imaging based workstation occupancy detection prototype for welding benches.

## Project Structure

- `dashboard/` - Web dashboard prototype for displaying workstation occupancy status.
- `sensor/` - Raspberry Pi and Lepton thermal camera scripts for frame capture, ROI analysis, and occupancy detection.

## Current Status

- Web dashboard prototype is implemented with simulated two-workstation data.
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

## Train First Neural-Network Baseline

The first training script uses a small NumPy MLP instead of Edge Impulse. This keeps the training process local, reproducible, and easy to explain in the dissertation. Edge Impulse can still be considered later for deployment, after the data pipeline and labels are validated.

```bash
python3 sensor/train_thermal_mlp.py \
  data/raw/20260611_195455 \
  data/raw/free_neighbor_person_01
```

The model is a one-hidden-layer neural network trained on the full 80x60 radiometric thermal frame. It outputs four classes:

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

## Run First Rule-Based Monitor

The first monitor keeps occupancy and thermal safety as separate state machines:

```bash
sudo python3 sensor/workstation_monitor.py
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
| `SAFE` | Tool-area temperature is below the safe threshold |
| `MONITORING` | The workstation is empty and warm; more trend data is needed |
| `COOLING` | Tool-area temperature is decreasing |
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
Unattended hot delay: 3 minutes
```

These are initial engineering values, not final research results. They should be calibrated from the labelled dataset.

The monitor also writes machine-readable status to:

```text
data/runtime/status.json
```

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
