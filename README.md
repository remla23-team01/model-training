[![Coverage Status](https://coveralls.io/repos/github/remla23-team01/model-training/badge.svg)](https://coveralls.io/github/remla23-team01/model-training)

# model-training

## Sentiment Analysis

This is a project to train a model that performs sentiment analysis on restaurant reviews.
The training pipeline is in `b1_Sentiment_Analysis_Model.ipynb`.
The inference pipeline is in `b2_Sentiment_Predictor.ipynb`.
Training data is in `a1_RestaurantReviews_HistoricDump.tsv`.

Credits
This project is based on Skillcate AI "Sentiment Analysis Project — with traditional ML & NLP".

To train the model run `python model-training.py1`, add the `--run_preprocessing` flag to rerun the preprocessing step.

## Running the pipeline

The pipeline is run whenever the code is pushed to the repository or a pull request is created. Once the pipeline has ran successfully,
it creates a coverage report and the pytest results.

## Requirements

### Create a virtual environment

Create a virtual enviroment to make sure the dependencies of each project is kept seperate. To create a virtual environment **venv**, run the following command:

```console
python3 -m venv venv
```

Activate the vitual environment venv using the following command:

```console
source ./venv/bin/activate
```

### Install the requirements

To download the project requirements, run the following command:

```console
python3 -m pip install -r requirements.txt
```

To update the requirements file, run the following command:

```console
python3 -m pip freeze > requirements.txt
```

## Running the project

## Running DVC

## Running the metrics

### Running tests with coverage

To run the tests with branch coverage, run the following command:

```console
pytest --cov-report term --cov-report xml:coverage.xml --cov
```

## Testing angles

The four testing angles are all implemented under the tests folder and are part of the pipeline.

## Mutamorphic testing

The mutamorphic testing is implemeted under the tests/test_mutamorphic.py and is part of the pipeline.

### Running pylint DSLint

To lint the files, run the following command:

```console
pylint --load-plugins=dslinter ./src
```
