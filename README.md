# Island Energy System Optimization Model

An optimization model for island energy infrastructure design based on Mixed Integer Linear Programming (MILP) and Gurobi solver. The system integrates multiple energy carriers with advanced reliability modeling and climate scenario analysis.

## 🌊 Project Overview

This project implements a optimization approach for island energy systems with the following features:
- Minimizes total system cost while ensuring reliable energy supply
- Models various energy technologies including renewables, storage, and conversion systems
- Employs Monte Carlo simulation for comprehensive reliability analysis
- Supports climate change impact assessment through future scenario modeling

### Core Features
- **Multi-Energy System**: Integrated electricity-heat-cold networks covering wind, solar, wave, LNG, and hydrogen energy sources
- **Advanced Reliability Modeling**: Monte Carlo simulation with K-means clustering for equipment failure handling
- **Climate Scenario Analysis**: Comparison between 2020 baseline and 2050 disaster scenarios, plus 2030-2050 future technology advancement scenarios
- **Optimization Engine**: Uses Gurobi MILP solver
- **Batch Processing**: Automated scripts for processing multiple island locations

## 🏗️ System Architecture

### Modeled Energy Technologies

**Renewable Energy:**
- Wind Turbines (WT)
- Solar Photovoltaic (PV)
- Wave Energy Converters (WEC)

**Conventional Systems:**
- LNG Storage and Consumption
- Combined Heat & Power (CHP)

**Energy Storage:**
- Electrical Storage Systems (ESS)
- Thermal Energy Storage (TES)
- Cold Energy Storage (CES)
- Hydrogen Storage (H2S)

**Energy Conversion:**
- Electric Boiler (EB)
- Air Conditioning (AC)
- Proton Exchange Membrane Electrolyzer (PEM)
- Fuel Cell (FC)
- LNG Vaporizer (LNGV)

## 📊 Scenario Analysis

The model supports multiple time scenarios with different climate projections and technology costs:

| Script File | Demand Baseline | Climate Disaster Data | Technology Cost Year | Purpose |
|-------------|-----------------|----------------------|---------------------|---------|
| `disaster_2020.py` | 2020 demand | 2020 climate | 2020 costs | 2020 baseline scenario optimization |
| `disaster_2050.py` | 2020 demand | 2050 climate | 2020 costs | System optimization under 2050 climate |
| `disaster_future_2030.py` | 2020 demand | 2050 climate | 2030 costs | 2030 technology cost projections |
| `disaster_future_2040.py` | 2020 demand | 2050 climate | 2040 costs | 2040 technology cost projections |
| `disaster_future_2050.py` | 2020 demand | 2050 climate | 2050 costs | 2050 technology cost projections |

All scenarios use **3-hour time resolution** for computational efficiency.

## 🚀 Quick Start

### System Requirements

**Required Python Packages:**
```bash
pip install pandas numpy xarray gurobipy scipy geopy pvlib timezonefinder scikit-learn matplotlib
```

**System Requirements:**
- Gurobi Optimizer (requires license)
- Recommended to run on computing clusters; for single island cases, suggest 16GB+ memory

### Basic Usage
**Batch Processing:**
```bash
# On Linux computing cluster terminal:
# Process all islands in chosen_island.csv for 2020/2050 scenarios
nohup ./run_jobs.sh > run_tasks.log 2>&1 &
# Process all islands in chosen_island.csv for technology advancement scenarios
nohup ./run_jobs_future.sh > run_tasks.log 2>&1 &
```

### Input Data Structure

The model requires the following data directories:

```
project_root/
├── demand/                    # Energy demand data
│   ├── demand_{lat}_{lon}.csv # Energy demand profiles
│   ├── pv_{lat}_{lon}.csv     # Solar generation profiles
│   └── wt_{lat}_{lon}.csv     # Wind generation profiles
├── CMIP6/                     # Climate model data
│   ├── MRI_2020_uas/         # 2020 wind data (u-component)
│   ├── MRI_2020_vas/         # 2020 wind data (v-component)
│   ├── MRI_2050_uas/         # 2050 wind projections
│   └── MRI_2050_vas/         # 2050 wind projections
├── wave/                      # Wave energy data
│   ├── wave_2020.nc          # 2020 wave energy
│   ├── waveheight_2020.nc    # 2020 wave heights
│   ├── wave_2050.nc          # 2050 wave projections
│   └── waveheight_2050.nc    # 2050 wave height projections
└── LNG/                       # LNG terminal location data
    └── LNG_Terminals.xlsx
```

## 📈 Model Formulation

### Optimization Objective
Minimize total system cost including:
- Capital expenditure (CAPEX) for all equipment
- Operational expenditure (OPEX)
- LNG transportation costs based on distance to nearest terminal
- Energy curtailment costs
- Load shedding costs

### Key Constraints
- **Energy Balance**: Supply-demand balance for all energy carriers
- **Reliability Requirements**: 99.9% availability (0.1% EENS limit)
- **Equipment Capacity**: Power and energy constraints for all technologies
- **Storage Operations**: Continuous charge/discharge with efficiency losses
- **LNG Procurement**: 14-day periodic purchasing cycles

### Reliability Modeling
1. **Monte Carlo Simulation**: Generate 1000 failure scenarios
2. **Equipment Failure Rates**: Based on wind speed and wave height thresholds
3. **K-means Clustering**: Automatic cluster selection using silhouette coefficient
4. **Repair Times**: Technology-specific downtime periods

## 📋 Output Files

Results are organized by scenario in dedicated output directories:

### Per-Island Results
- `{lat}_{lon}_best_cost.csv` - Detailed cost breakdown
- `{lat}_{lon}_capacity.csv` - Optimal equipment capacities
- `{lat}_{lon}_results.csv` - Hourly operational results for complete time horizon

### Batch Processing Logs
- `logs/main_log.log` - Master execution log for 2020/2050 climate scenarios
- `logs/main_future_log.log` - Master execution log for future scenarios
- `logs/gap_failure_islands.csv` - Islands that didn't meet 1% MIP gap target

## 🔬 Research Applications

This model supports research in:
- **Climate Change Impact Assessment**: Comparing energy system resilience across climate scenarios
- **Energy Security Analysis**: Quantifying reliability improvements from different technologies
- **Economic Optimization**: Cost-benefit analysis of renewable vs. conventional systems
- **Policy Planning**: Infrastructure investment strategies for remote communities

---

**Note**: For the Chinese version of this README, see [README_CN.md](README_CN.md).