# NNIA final Project
The repository contains necessary scripts and data concerning NNIA final project. Goals:
-   Part 1
    - Data pre-processing
    - extract relevant information e.g. POS tags
    - Analyze: size, classes, balanced/imbalanced, length of sequences​
-   Part 2
    -   Encode data using BERT​
    -   Train a LSTM model for POS​
    -   Track training performance using wandb


## Table of Contents
1.  `data`
    -   sample.conll
2.  `ontonetes-4.0`
3.  `results`
4.  `src`
    -   data_preprocess.py
    -   ontonotes4.py
    -   run.py
    -   tokenizze.py
5.  `environment.yml`
6.  instructions
    -   instructions_part1_1.pdf
    -   instructions_part1_2.pdf
7.  LICENSE
8.  README.md

## General information
-   Prepare the envrionment to use:
    ```
    conda env update --file environment.yml
    ```

## Usage 
### Data Preprocessing
#### Part 1
-   **Help**: for instructions on how to run the script with appropriate arguments.\
    `python src/data_preprocess.py --help`

    ```
    usage: data_preprocess.py [-h] input_f out_dir

    Data-preprocessing script

    positional arguments:
    input_f     path to input file in conll format
    out_dir     output dir to save results

    optional arguments:
    -h, --help  show this help message and exit```
    
- **Run pre-processing**
    ```
    python src/data_preprocess.py data/sample.conll results
    ```
#### Part 2
- **Load the dataset and tokenize**
    `python src/tokenizze.py ontonetes-4.0/`
    ```
    usage: tokenizze.py [-h] [-save_emb SAVE_EMB] [-load_emb LOAD_EMB] [-batch_size BATCH_SIZE] ip_dir

    Tokenization and generating BERT embeddings using the transformers Package

    positional arguments:
    ip_dir                Ontonotes4 dir with conll gold files

    optional arguments:
    -h, --help            show this help message and exit
    -save_emb SAVE_EMB    save BERT embeddings
    -load_emb LOAD_EMB    choice to load pre-trained embeddings
    -batch_size BATCH_SIZE
                            batch size when generating embedding
    ```
- **Train and Test**

    LSTM

    `python src/run.py lstm models/`

    NOTE: GRU is prone to errors at the moment.
    ```
    usage: run.py [-h] [-num_layers NUM_LAYERS] [-dropout DROPOUT] [-batch_size BATCH_SIZE] [-hidden_dim HIDDEN_DIM] [-epochs EPOCHS] model_choice emb_dir

    positional arguments:
    model_choice          choose your model: lstm, gru
    emb_dir               path to embeddings dir

    optional arguments:
    -h, --help            show this help message and exit
    -num_layers NUM_LAYERS
                            number of hidden layers
    -dropout DROPOUT      dropout parameter
    -batch_size BATCH_SIZE
                            batch size
    -hidden_dim HIDDEN_DIM
                            dimension of hidden layers
    -epochs EPOCHS        number of training epochs
    ```
- **Results**
    Training performance can be seen at https://wandb.ai/sangeet2020/LSTM%20Bert%20POS%20tagging?workspace=user-sangeet2020

