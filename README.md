# valorant-match-predictor
I like valorant!

## Creating Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```

## Downloading Dependencies
```
pip3 install -r requirements.txt
```

## Updating Dependencies
```
pip3 freeze > requirements.txt
```

## Formatting
```
black .
```

## Deleting Virtual Environment
```
deactivate
rm -rf venv
```

## Running Neural Network
```
python3 main.py
```

## Dataset
https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data
#### NOTE: rename the dataset folder from 'archive' to 'data'