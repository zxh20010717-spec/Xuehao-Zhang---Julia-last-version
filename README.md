# BeijingPM25Prediction.jl

A Julia package for predicting Beijing PM2.5 concentrations using Neural Ordinary Differential Equations (Neural ODEs).

## Overview

This package implements a data-driven approach for air quality prediction using the formulation:

`
dC/dt = MLP(C, W)
`

Where:
- C: PM2.5 concentration
- W: Weather conditions (temperature, pressure, dew point)  
- MLP: Multi-layer perceptron neural network

## Features

- 🎯 Multi-horizon prediction (1 hour, 1 day, 1 week)
- 📊 Comprehensive performance evaluation (MAE, RMSE, R²)
- 📈 Automatic visualization generation
- 🔧 Modular and extensible design
- ✅ Full test coverage

## Installation

`julia
# Clone the repository
git clone https://github.com/yourusername/BeijingPM25Prediction.jl.git
cd BeijingPM25Prediction.jl

# Activate the project environment
julia --project=.

# Install dependencies
julia> using Pkg; Pkg.instantiate()
`

## Quick Start

### Data Preparation
1. Download the Beijing Multi-Site Air Quality Dataset
2. Place CSV files in the data/raw/ directory

### Run Experiment
`julia
using BeijingPM25Prediction

# Run complete experiment
results = run_pm25_experiment()

# Analyze results
analyze_results(results)
`

### Command Line Usage
`ash
julia --project=. scripts/run_experiment.jl
`

## Results

The Neural ODE approach achieves:
- **1-hour prediction**: R² ≈ 0.95, MAE ≈ 16 μg/m³
- **24-hour prediction**: R² ≈ 0.18, MAE ≈ 59 μg/m³  
- **168-hour prediction**: R² ≈ 0.07, MAE ≈ 173 μg/m³

## Project Structure

`
BeijingPM25Prediction/
├── src/
│   └── BeijingPM25Prediction.jl    # Main module
├── test/
│   └── runtests.jl                 # Test suite
├── scripts/
│   └── run_experiment.jl           # Experiment runner
├── data/
│   └── raw/                        # Raw data files
├── results/
│   └── figures/                    # Generated plots
├── docs/                           # Documentation
├── Project.toml                    # Package configuration
└── README.md                       # This file
`

## Citation

If you use this package in your research, please cite:

`ibtex
@misc{BeijingPM25Prediction,
  title={Data-Driven Neural ODE Approach for Beijing PM2.5 Prediction},
  author={Student Name},
  year={2025},
  url={https://github.com/yourusername/BeijingPM25Prediction.jl}
}
`

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Beijing Municipal Environmental Monitoring Center for data
- UCI Machine Learning Repository for data distribution
- Julia community for excellent scientific computing ecosystem
