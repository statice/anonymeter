# Anonymeter: Unified Framework for Quantifying Privacy Risk in Synthetic Data

`Anonymeter` is a unified statistical framework to jointly quantify different
types of privacy risks in synthetic tabular datasets. `Anonymeter` is equipped
with attack-based evaluations for the **Singling Out**, **Linkability**, and
**Inference** risks, which are the three key indicators of factual anonymization
according to the [Article 29 Working Party](https://ec.europa.eu/justice/article-29/documentation/opinion-recommendation/files/2014/wp216_en.pdf).

A throughout description of the working of the framework and the attack
algorithms can be found in the paper
[A Unified Framework for Quantifying Privacy Risk in Synthetic Data](https://arxiv.org/abs/2211.10459).
This work has been accepted at the 23rd Privacy Enhancing
Technologies Symposium ([PETS 2023](https://petsymposium.org/cfp23.php)).


## Usage

### Downloading Code

Clone the Anonymeter repository:

```shell
git clone git@github.com:statice/anonymeter.git
```

### Installation

Install  [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
After installing Miniconda, create an environment:

```shell
conda create -n anonymeter python=3.9
conda activate anonymeter
```

Once you have this basic environment set up, you can install this package.
Use the `-e` option to install in editable mode.

```shell
cd anonymeter
pip install -e '.[build,dev]'
```

### Getting started

Check out the example notebook in the `notebooks` folder to start playing around
with `anonymeter`.

## Cite this work

If you use anonymeter in your work, we would appreciate citations to the following paper:

"A Unified Framework for Quantifying Privacy Risk in Synthetic Data", M. Giomi *et al*, PoPETS 2023.

This `bibtex` entry can be used to refer to the arxiv preprint:

```text
@misc{https://doi.org/10.48550/arxiv.2211.10459,
  doi = {10.48550/ARXIV.2211.10459},
  url = {https://arxiv.org/abs/2211.10459},
  author = {Giomi, Matteo and Boenisch, Franziska and Wehmeyer, Christoph and Tasnádi, Borbála},
  title = {A Unified Framework for Quantifying Privacy Risk in Synthetic Data},
  publisher = {arXiv},
  year = {2022}
}
```
