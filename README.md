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

Useful options:

```bash
sudo python3 sensor/thermal_roi_viewer.py --threshold 30 --scale 10
```

## Planned Pipeline

```text
FLIR Lepton -> Raspberry Pi -> Python/OpenCV ROI analysis -> Occupancy status -> Web dashboard
```
