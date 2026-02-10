# Estimation-based-Formation-Control-with-Orientation-Alignment

This repository contains the simulation source code for estimation-based formation control with orientation alignment. It implements the proposed method alongside comparison baseline methods as described in our paper.

---

## Video

Click the image below to watch on Youtube.

[![Simulation Video](https://img.youtube.com/vi/0VggcP6UHuI/0.jpg)](https://www.youtube.com/watch?v=0VggcP6UHuI)

> **Note:** Video will be updated later.

---

## Paper

Under review, will be updated later.

---

## Overview

The project is expansion of series of Multi-Agent Systems (MAS) in **HMCL** and aims to provide a comprehensive framework for estimation-based formation control with orientation alignment.

---

## Getting Started

### Prerequisites
Ensure you have Anaconda installed on your system.

```bash
conda create -n myenv python=3.10.19
conda activate myenv
pip install numpy matplotlib scipy pandas numba cvxpy control tqdm openpyxl
```


### Installation
Clone this repository to your local machine using:

```bash
git clone https://github.com/HMCL-UNIST/Estimation-based-Formation-Control-with-Orientation-Alignment.git
```


### Running the Simulation
To run the simulation, execute the following command in the terminal:

```bash
python3 MCMC_comparison.py
```


### Configuration Options
Before running the simulation, you can configure several parameters within MCMC_comparison.py to tailor the simulation to your needs:

1. **Number of Monte Carlo Runs**: Adjust the number of repetitions for each scenario by setting the "num_runs_per_scenario" variable. This ensures the statistical reliability of the results. For example: "num_runs_per_scenario = 20".

2. **Network Connectivity**: Specify the interaction range for each agent by defining the "connectivity_values" list. Each value represents the number of neighbors an agent interacts with in a directed ring topology. For instance: "connectivity_values = list(range(2, 10))"

3. **Parallel Processing Workers**: Specify the maximum number of worker threads to be used for concurrent simulations by defining "max_concurrent_processes". Increasing this value can significantly reduce total execution time on multi-core CPUs. For instance: "max_concurrent_processes = 2".


### Configuration Options
To check the RMSE, execute the following command in the terminal:

```bash
python3 RMSE_scatter.py
```
