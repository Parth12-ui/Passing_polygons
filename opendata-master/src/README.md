# 📦 Source Code (src)

This directory contains reusable Python modules for loading, processing, and visualizing SkillCorner data.

## 🏗️ Structure

- **`data/`**: Scripts for data ingestion and loading.
  - `basic_loading.py`: Functions to load match metadata and tracking data.
- **`features/`**: Feature engineering and aggregation logic.
  - `DynamicEventsAggregator.py`: Logic for summarizing dynamic event categories.
  - `PhasesOfPlayAggregator.py`: Framework for aggregating data by game phases.
- **`visualization/`**: Reusable plotting and reporting functions.
  - `head2head_viz.py`: High-fidelity visualization for team comparisons.

## 🛠️ Usage

These modules are designed to be imported into tutorials or custom scripts:

```python
from src.features.PhasesOfPlayAggregator import PhasesOfPlayAggregator
# ... initialize and use ...
```
