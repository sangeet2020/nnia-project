#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   1970-01-01 01:00:00
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlande
# @Last Modified time: 2021-03-20 20:25:48


"""
<Function of script>
"""

import os
import sys
import argparse
import numpy as np
from transformers import AutoTokenizer, BertModel
import pdb
import datasets
import torch
from collections import defaultdict

MAX_SEQ_LENGTH = 64
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def my_function(arg_1, arg_2, args):
    """ purpose of my function """

def smart_split(sentences, pos_tags, max):
    new_sents=[]
    new_tags=[]
    for data in sentences:
        new_sents.append(([data[x:x+max] for x in range(0, len(data), max)]))
    new_sents = [val for sublist in new_sents for val in sublist]
    
    for data in pos_tags:
        new_tags.append(([data[x:x+max] for x in range(0, len(data), max)]))
    new_tags = [val for sublist in new_tags for val in sublist]
    
    return new_sents, new_tags


def bert_labels(labels):
    train_label_bert = []
    train_label_bert.append('-PAD-')
    for i in labels:
        train_label_bert.append(i)
    train_label_bert.append('-PAD-')
    print('BERT labels:', train_label_bert)


def main():
    """ main method """
    args = parse_arguments()
    # os.makedirs(args.out_dir, exist_ok=True)
    dataset = datasets.load_dataset('ontonotes4.py', data_files='../data/sample.conll')
    
    # Determine maximum sequence length
    train_sents = [item["token"] for item in dataset["train"]["triplet"]]
    train_tags = [item["pos_tag"] for item in dataset["train"]["triplet"]]
    
    test_sents = [item["token"] for item in dataset["test"]["triplet"]]
    test_tags = [item["pos_tag"] for item in dataset["test"]["triplet"]]
    
    valid_sents = [item["token"] for item in dataset["validation"]["triplet"]]
    valid_tags = [item["pos_tag"] for item in dataset["validation"]["triplet"]]
    
    # pdb.set_trace()
    print('Max sentence length:', max(len(max(train_sents, key=len)), 
                                    len(max(test_sents, key=len)), 
                                    len(max(valid_sents, key=len))))
    
    train_sents, train_tags = smart_split(train_sents, train_tags, MAX_SEQ_LENGTH)    
    test_sents, test_tags = smart_split(test_sents, test_tags, MAX_SEQ_LENGTH)
    valid_sents, valid_tags = smart_split(valid_sents, valid_tags, MAX_SEQ_LENGTH)

    print('Max sentence length:', max(len(max(train_sents, key=len)), 
                                    len(max(test_sents, key=len)), 
                                    len(max(valid_sents, key=len))))
    
    """
    Tokenization using the transformers Package
    1. Tokenize the input sentence
    2. Add the [CLS] and [SEP] tokens.
    3. Pad or truncate the sentence to the maximum length allowed
    4. Encode the tokens into their corresponding IDs Pad or truncate all sentences to the same length.
    5. Create the attention masks which explicitly differentiate real tokens from [PAD] tokens
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    # return_tensors = 'pt'
    data = defaultdict()
    for split in dataset:
        for id, sentence in enumerate(dataset[split]["raw"]):
            
            if split not in data:
                data[split] = {id: tokenizer(sentence,
                                            truncation=True,
                                            pad_to_max_length=True,
                                            max_length = MAX_SEQ_LENGTH
                )}
            else:
                data[split].setdefault(id, tokenizer(sentence,
                                                    truncation=True, 
                                                    pad_to_max_length=True, 
                                                    max_length = MAX_SEQ_LENGTH
                                                    ))
            # pdb.set_trace()
    encoded_tokens = [item["input_ids"] for id, item in data["train"].items()]
    model = BertModel.from_pretrained("bert-base-cased", 
                                    output_hidden_states = True)
    model.to(DEVICE)
    model.eval()
    # pdb.set_trace()
    with torch.no_grad():
        outputs = model(torch.LongTensor(encoded_tokens).to(DEVICE))

    pdb.set_trace()
    
    # for i,v in data["train"].items():print(len(v["input_ids"][1]))
    # # One-hot encode labels
    # train_labels = to_categorical(train_labels_ids, num_classes=n_tags)
    # test_labels = to_categorical(test_labels_ids, num_classes=n_tags)
    
    pdb.set_trace()

    
    pdb.set_trace()

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("arg_1", help="describe arg_1")
    # parser.add_argument("arg_2", help="describe arg_2")
    # parser.add_argument("-optional_arg", default=default_value, type=int/"", help='optional_arg meant for some purpose')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()