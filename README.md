# Sentiment Analysis

This is a project to train a model that performs sentiment analysis on restaurant reviews. The repo follows the [cookiecutter](https://github.com/drivendata/cookiecutter-data-science) format.
It is set up using DVC and a google drive remote.

To train the model, clone the branch and install all dependencies in the `requirements.txt`
Then you should be able to run `dvc repro` to train the model and look at the metrics.
To check metrics of different branches use `dvc metrics diff`

The model training is split up in 3 parts. The collecting of the data, the preprocessing and finaly the model training. Each step has its own outputs that are then used for the next step. 


### Credits

This project is based on Skillcate AI "Sentiment Analysis Project — with traditional ML & NLP".