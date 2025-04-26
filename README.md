# valorant-match-predictor
I like valorant!

## How to Run
1) Install pipx: [https://pipx.pypa.io/stable/](https://pipx.pypa.io/stable/)
    - **For Linux/WSL:**
    ```
    sudo apt update
    
    sudo apt install pipx

    pipx --version
    ```
    - **For macOS:**
    ```
    brew install pipx

    pipx ensurepath

    pipx --version
    ```

2) Download poetry: https://python-poetry.org/docs/#installing-with-pipx
    ```
    pipx install poetry==1.8.2

    poetry --version
    ```

3) Launch poetry shell
    ```
    poetry shell

    poetry install
    ```

4) Install dependencies
    ```
    make install
    ```

5) Running the various programs
    ```
    make model
    ```
    ```
    make scrape
    ```

## Updating Dependencies
```
poetry add libraryname
```

## Formatting
```
black .
```

## Dataset
https://www.kaggle.com/datasets/ryanluong1/valorant-champion-tour-2021-2023-data