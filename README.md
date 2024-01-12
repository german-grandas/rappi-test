# Rappi test

This project includes code for the rappi ml test.

## Create a custom configuration
The project are based on yml configurations, where you can update of modify the parameters to the training process 
and create new experiments.

Example:
```yaml
DATA:
  TYPE: "TitanicDataset"
  PATH: "./data"
  OBJECTIVE_VARIABLE: "Survived"
MODEL:
  NAME: "StatsModel"
  ARCHITECTURE: "sm.Logit"
  SUMMARY: True
EVALUATION:
  METRICS: ["f1", "specificity", "recall"]
```
## Installation

Make sure you have [Python > 3.11](https://www.python.org/) and [Poetry](https://python-poetry.org/) installed.

```bash
# Install dependencies
poetry install
```
## Build package 

```bash
    poetry build
```
## Installation of the package
```bash
    poetry add .whl_file
```
or
```bash
    pip install .whl_file
```
## Usage
To use the cli interface:

### Train
```bash
    rappi-ml train --config-path=./path/to/configuration.yml
```

### Inference
```bash
    rappi-ml inference --config-path=./path/to/configuration.yml --data=Some_data
```

## Running tests
```bash
poetry run pytest
```