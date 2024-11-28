# Fake-News
Uni project to create a classifier to detect fake news

# Setup the environment

```commandline
conda create -n fake-news python=3.12
conda activate fake-news
pip install -r requirements.txt
python -m spacy download en
```

To export new packages to .txt file:
```commandline
pip freeze > requirements.txt
```

Please don't push data to the repository, instead create a jupyter notebook to download the data.
For Bert usage please install compatible torch version from https://pytorch.org