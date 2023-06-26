[![Coverage Status](https://coveralls.io/repos/github/remla23-team01/model-training/badge.svg)](https://coveralls.io/github/remla23-team01/model-training)

# Model-training

## Sentiment Analysis
This is a project to train a model that performs sentiment analysis on restaurant reviews.

#### Credits
This project is based on Skillcate AI "Sentiment Analysis Project — with traditional ML & NLP".

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
The training of the model is setup using DVC. Training is split up in the following 3 steps, getting the data, preprocessing the data and finally training the model. The python scripts for each step can be found in the `/src` folder. `get_data.py` downloads the data and saves it locally. `preprocess.py` takes this file and performs some preprocessing steps on the data and saves both the preprocessed data and the preprocessing object. Finally, `train_model.py` trains the model on the processed data and saves the model and some metrics.

Running these files in the same order as explained will train the model.

## Run using DVC
All of these output files are tracked using DVC to prevent unneccesary steps from being executed. `dvc.yaml` clearly shows all the input, output and metrics files. DVC tracks these files and stores them in a google drive folder.

To pull files from the remote drive folder run:

```
dvc pull
```

To run the model training, run the following command:
```
dvc repro
```

This will run each of the steps, but only if something changed compared to previous runs.

## Running the metrics
To look at the accuracy of the model run: 
```
dvc metrics show
```

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
