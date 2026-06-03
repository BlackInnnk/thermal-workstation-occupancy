# Thermal Workstation Occupancy

A thermal imaging based workstation occupancy detection prototype for welding benches.

## Project Structure

- `dashboard/` - Web dashboard prototype for displaying workstation occupancy status.
- `sensor/` - Raspberry Pi and Lepton thermal camera scripts for frame capture, ROI analysis, and occupancy detection.

## Current Status

- Web dashboard prototype is implemented with simulated two-workstation data.
- Lepton thermal camera capture has been tested with `pylepton`.
- Sensor-side ROI detection scripts will be added under `sensor/`.

## Planned Pipeline

```text
FLIR Lepton -> Raspberry Pi -> Python/OpenCV ROI analysis -> Occupancy status -> Web dashboard
```
