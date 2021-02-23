# NNIA final Project
The repository contains necessary scripts and data concerning NNIA final project. Goals:
-   Part 1
    - Data pre-processing
    - extract relevant information e.g. POS tags
    - Analyze: size, classes, balanced/imbalanced, length of sequences​
-   (TODO) Part 2
    -   Encode data using BERT​
    -   Train a model for POS​
    -   Choose hyperparameters using wandb


## Table of Contents
1.  `data`
    -   sample.conll
2.  `results`
3.  `src`
    -   data_preprocess.py
4.  `environment.yml`
5.  instructions
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

### Training
-   TODO

