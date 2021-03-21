#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   1970-01-01 01:00:00
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlande
# @Last Modified time: 2021-03-20 20:25:48


"""
Tokenization and generating BERT embeddings using the transformers Package
"""

import os
import sys
import pdb
import torch
import pickle
import logging
import argparse
import datasets
import subprocess
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, BertModel

MAX_SEQ_LENGTH = 32
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def main():
    """ main method """
    args = parse_arguments()
    root = subprocess.Popen("git rev-parse --show-toplevel", shell=True, stdout=subprocess.PIPE)
    TOP_DIR = root.stdout.read()
    TOP_DIR = TOP_DIR.strip().decode("utf-8")
    
    print('-'*20+"Loading dataset"+'-'*20)
    dataset = datasets.load_dataset(TOP_DIR+'/src/ontonotes4.py', data_files=args.ip_dir)

    # Determine maximum sequence length
    train_sents = [item["token"] for item in dataset["train"]["triplet"]]
    train_tags = [item["pos_tag"] for item in dataset["train"]["triplet"]]
    
    test_sents = [item["token"] for item in dataset["test"]["triplet"]]
    test_tags = [item["pos_tag"] for item in dataset["test"]["triplet"]]
    
    valid_sents = [item["token"] for item in dataset["validation"]["triplet"]]
    valid_tags = [item["pos_tag"] for item in dataset["validation"]["triplet"]]
    
    print('Max sentence length:', max(len(max(train_sents, key=len)), 
                                    len(max(test_sents, key=len)), 
                                    len(max(valid_sents, key=len))))
    
    # Tokenization using the transformers Package
    print('-'*20+"Tokenizing dataset"+'-'*20)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    data = defaultdict()
    for split in dataset:
        for id, sentence in enumerate(dataset[split]["raw"]):
            if split not in data:
                data[split] = {id: tokenizer(sentence,
                                            truncation=True,
                                            padding=True,
                                            max_length=MAX_SEQ_LENGTH,
                                            return_tensors='pt'
                )}
            else:
                data[split].setdefault(id, tokenizer(sentence,
                                                    truncation=True, 
                                                    padding=True, 
                                                    max_length=MAX_SEQ_LENGTH,
                                                    return_tensors='pt'
                                                    ))
    print("--Done--")
    
    # Generating BERT embeddings for the sequence of tokens
    out_dir = TOP_DIR + "/models/"
    
    if args.load_emb:
        if os.path.isfile(out_dir + "embeddings_ontonotes.pkl"):
            print('-'*20+"Pretrained embeddings found."+'-'*20)
            print("Loading...")
            infile = open(args.load_emb,'rb')
            sents_embeddings = pickle.load(infile)
            infile.close()
        else:
            print("Error. No embeddings found")
        
    else:
        print('-'*20+"Generating embeddings"+'-'*20)
        model = BertModel.from_pretrained("bert-base-cased",  output_hidden_states=True)
        model.to(DEVICE)
        model.eval()

        sents_embeddings = []
        encoded_tokens = [item["input_ids"] for id, item in data["train"].items()]
        with torch.no_grad():
            for  item in encoded_tokens:
                outputs = model(torch.LongTensor(item).to(DEVICE), output_hidden_states=True)
                embed = outputs.hidden_states[0]
                    
                sents_embeddings.append({
                    'sent':item,
                    "embedding": embed
                })
                
        # Dump embeddings
        pdb.set_trace()
        if args.save_emb:
            os.makedirs(out_dir, exist_ok=True)
            outfile = open(out_dir+"embeddings_ontonotes.pkl",'wb')
            pickle.dump(sents_embeddings, outfile)
            outfile.close()
            print("Embeddings saved at "+str(out_dir+"embeddings_ontonotes.pkl"))
    # print(sents_embeddings[0]['sent'].size())
    # print(sents_embeddings[0]['embedding'].size())
    
    print("--Done--")

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ip_dir", help="Ontonotes4 dir with conll gold files")
    parser.add_argument("-save_emb", default=True, type=bool, help='save BERT embeddings')
    parser.add_argument("-load_emb", default=False, type=bool, help='choice to load pre-trained embeddings')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()