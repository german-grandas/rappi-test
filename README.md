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

## Runinng coverage
```bash
poetry run coverage report -m
```
### Coverage report
```bash
Name                                     Stmts   Miss  Cover   Missing
----------------------------------------------------------------------
source\__init__.py                           0      0   100%
source\datasets\__init__.py                  2      0   100%
source\datasets\titanic.py                  36      2    94%   14, 34
source\inference.py                         16      2    88%   8, 15
source\model\__init__.py                     3      0   100%
source\model\evaluation.py                  22      0   100%
source\model\stats_model.py                 16      0   100%
source\train.py                             23      2    91%   8, 15
tests\__init__.py                            0      0   100%
tests\datasets\__init__.py                   0      0   100%
tests\datasets\titanic_dataset_test.py      31      0   100%
tests\inference_test.py                     18      0   100%
tests\model\__init__.py                      0      0   100%
tests\model\evaluation_test.py              22      0   100%
tests\model\stats_model_test.py             20      0   100%
tests\train_test.py                         24      0   100%
----------------------------------------------------------------------
TOTAL                                      233      6    97%
```