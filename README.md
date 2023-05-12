# Anonymeter: Unified Framework for Quantifying Privacy Risk in Synthetic Data

`Anonymeter` is a unified statistical framework to jointly quantify different
types of privacy risks in synthetic tabular datasets. `Anonymeter` is equipped
with attack-based evaluations for the **Singling Out**, **Linkability**, and
**Inference** risks, which are the three key indicators of factual anonymization
according to the [Article 29 Working Party](https://ec.europa.eu/justice/article-29/documentation/opinion-recommendation/files/2014/wp216_en.pdf).


> Anonymeter has been positively reviewed by the technical experts from the [Commission Nationale de l’Informatique et des Libertés (CNIL)](https://www.cnil.fr/en/home) which, in their words, _“have not identified any reason suggesting that the proposed set of methods could not allow to effectively evaluate the extent to which the aforementioned three criteria are fulfilled or not in the context of production and use of synthetic datasets”_. The CNIL also expressed the opinion that the results of Anonymeter (i.e. the three risk scores) **should be used by the data controller to decide whether the residual risks of re-identification are acceptable or not, and whether the dataset could be considered anonymous**.


## `Anonymeter` in a nutshel

In `Anonymeter` each privacy risk is derived from a privacy attacker whose task is to use the synthetic dataset
to come up with a set of *guesses* of the form:
- "there is only one person with attributes X, Y, and Z" (singling out)
- "records A and B belong to the same person" (linkability)
- "a person with attributes X and Y also have Z" (inference)

Each evaluation consists of running three different attacks:
- the "main" privacy attack, in which the attacker uses the synthetic data to guess information on records in the original data.
- the "control" privacy attack, in which the attacker uses the synthetic data to guess information on records in the control dataset.
- the "baseline" attack, which models a naive attacker who ignores the synthetic data and guess randomly.

Checking how many of these guesses are correct, the success rates of the different attacks are measured and used to
derive an estimate of the privacy risk. In particular, the "control attack" is used to separate what the attacker
learns from the *utility* of the synthetic data, and what is instead indication of privacy leaks.
The "baseline attack" instead functions as a sanity check. The "main attack" attack should outperform random
guessing in order for the results to be trusted.

For more details, a throughout
description of the framework and the attack algorithms can be found in the paper
[A Unified Framework for Quantifying Privacy Risk in Synthetic Data](https://petsymposium.org/popets/2023/popets-2023-0055.php), accepted at the 23rd Privacy Enhancing Technologies Symposium ([PETS 2023](https://petsymposium.org/cfp23.php)).



## Setup and installation

`Anonymeter` requires Python 3.8.x, 3.9.x or 3.10.x installed. The simplest way to install `Anonymeter` is from `PyPi`. Simply run

```
pip install anonymeter
```

and you are good to go.

### Local installation

To install `Anonymeter` locally, clone the repository:

```shell
git clone git@github.com:statice/anonymeter.git
```

and install the dependencies:

```shell
cd anonymeter  # if you are not there already
pip install . # Basic dependencies
pip install ".[notebooks]" # Dependencies to run example notebooks
pip install -e ".[notebooks,dev]" # Development setup
```

If you experience issues with the installation, we recommend to install
`anonymeter` in a new clean virtual environment.

## Getting started

Check out the example notebook in the `notebooks` folder to start playing around
with `anonymeter`. To run this notebook you would need `jupyter` and some plotting libraries.
This should be installed as part of the `notebooks` dependencies. If you haven't done so, please
install them by executing:

```shell
pip install anonymeter[notebooks]
```
if you are installing anonymeter from `PyPi`, or:

```shell
pip install ".[notebooks]"
```

if you have opted for a local installation.

## Basic usage pattern

For each of the three privacy risks anonymeter provide an `Evaluator` class. The high-level classes `SinglingOutEvaluator`, `LinkabilityEvaluator`, and `InferenceEvaluator` are the only thing that you need to import from `Anonymeter`.

Despite the different nature of the privacy risks they evaluate, these classes have the same interface and are used in the same way. To instantiate the evaluator you have to provide three dataframes: the original dataset `ori` which has been used to generate the synthetic data, the synthetic data `syn`, and a `control` dataset containing original records which have not been used to generate the synthetic data.

Another parameter common to all evaluators is the number of target records to attack (`n_attacks`). A higher number will reduce the statistical uncertainties on the results, at the expense of a longer computation time.

```python
evaluator = *Evaluator(ori: pd.DataFrame,
                       syn: pd.DataFrame,
                       control: pd.DataFrame,
                       n_attacks: int)
```

Once instantiated the evaluation pipeline is executed when calling the `evaluate`, and the resulting estimate of the risk can be accessed using the `risk()` method.

```python
evaluator.evaluate()
risk = evaluator.risk()
```

## Configuring logging

`Anonymeter` uses the standard Python logger named `anonymeter`.
You can configure the logging level and the output destination
using the standard Python logging API (see [here](https://docs.python.org/3/library/logging.html) for more details).

For example, to set the logging level to `DEBUG` you can use the following snippet:

```python
import logging

# set the logging level to DEBUG
logging.getLogger("anonymeter").setLevel(logging.DEBUG)
```

And if you want to log to a file, you can use the following snippet:

```python
import logging

# create a file handler
file_handler = logging.FileHandler("anonymeter.log")

# set the logging level for the file handler
file_handler.setLevel(logging.DEBUG)

# add the file handler to the logger
logger = logging.getLogger("anonymeter")
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)
```


## Cite this work

If you use anonymeter in your work, we would appreciate citations to the following paper:

"A Unified Framework for Quantifying Privacy Risk in Synthetic Data", M. Giomi *et al*, PoPETS 2023.
This `bibtex` entry can be used to refer to the paper:

```text
@misc{anonymeter,
  doi = {https://doi.org/10.56553/popets-2023-0055},
  url = {https://petsymposium.org/popets/2023/popets-2023-0055.php},
  journal = {Proceedings of Privacy Enhancing Technologies Symposium},
  year = {2023},
  author = {Giomi, Matteo and Boenisch, Franziska and Wehmeyer, Christoph and Tasnádi, Borbála},
  title = {A Unified Framework for Quantifying Privacy Risk in Synthetic Data},
}
```
