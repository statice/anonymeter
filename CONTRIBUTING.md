# Contribution Guide

## Releasing a new version

### Building the new version

Increment the version in `pyproject.toml`.

Install development dependencies:
```bash
pip install ".[dev]"
```

Build a source and a wheel distributions:
```bash
rm -rf ./dist # clean the build directory if necessary
python -m build
```

### Publishing to Test Pypi

Login to Test PyPi. Create a new account if you don't have it yet
and ask to be added as a collaborator for Anonymeter.

Get the token from [Test PyPi](https://test.pypi.org/manage/account/#api-tokens)
and save it as suggested to `$HOME/.pypirc`:
```toml
[testpypi]
  username = __token__
  password = YOUR_TOKEN_HERE
```

Upload the artifacts to Test PyPi:
```bash
twine upload --repository testpypi dist/*
```

Test that the package installs and works properly. For example,
you can create a new virtualenv and try to install the package there.
```bash
mkdir ~/test-anonymeter # create some test directory
cd ~/test-anonymeter
python -m venv .venv # create new virtual env
source .venv/bin/activate
asdf reshim python # in case you use asdf
pip install --upgrade pip
pip install --index-url https://test.pypi.org/simple anonymeter==NEW_VERSION
```

You can check that anonymeter is working by running it against the original tests.
For example, if you had Anonymeter repository checked out in `~/code/anonymeter`::
```
ln -s ~/code/anonymeter/tests ~/test-anonymeter/tests
pip install pytest
python -m pytest
```

### Publishing to PyPi

Once you tested the package with Test PyPi, you're ready to publish to
the original PyPi.

Login to PyPi. Create a new account if you don't have it yet
and ask to be added as a collaborator for Anonymeter.

Get the token from PyPi: https://pypi.org/manage/account/token
and add it as suggested to `$HOME/.pypirc`:
```toml
[pypi]
  username = __token__
  password = YOUR_TOKEN_HERE
```

Upload the artifacts to PyPi:
```bash
twine upload dist/*
```

### Publish the new release to GitHub

Commit the version update:

```bash
git commit -m "Version bump -> NEW_VERSION" pyproject.toml
```

Tag the source: `git tag NEW_VERSION`.

Push commit and tag: `git push --tags origin main`.
