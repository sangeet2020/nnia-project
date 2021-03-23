#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Zena Al Khalili, Sangeet Sagar
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
import time
import logging
import argparse
import datasets
import subprocess
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, BertModel

MAX_SEQ_LENGTH = 64 # probably unused
# DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")

def smart_split(sentences, pos_tags, max):
    """Smart split makes sure, when you truncate input sequence you dont loose data.
    To do this, it breaks the input sequence upto MAX_SEQ_LENGTH and the reaming part of the sequence 
    becomes a new sequence.
    For example, if MAX_SEQUENCE_LENGTH=64, a sentence with length 150 splits in 3 sentences: 150=64+64+22
    Args:
        sentences ([type]): [description]
        pos_tags ([type]): [description]
        max ([type]): [description]
    Returns:
        [type]: [description]
    """
    new_sents=[]
    new_tags=[]
    for data in sentences:
        new_sents.append(([data[x:x+max] for x in range(0, len(data), max)]))
    new_sents = [val for sublist in new_sents for val in sublist]
    
    for data in pos_tags:
        new_tags.append(([data[x:x+max] for x in range(0, len(data), max)]))
    new_tags = [val for sublist in new_tags for val in sublist]
    
    return new_sents, new_tags

def load_text_batchwise(data, args):
    batch_size = args.batch_size
    l = len(data)
    for ndx in range(0, l, batch_size):
        yield data[ndx:min(ndx + batch_size, l)]
        

def select_split(dataset, split):
    sents = [item["token"] for item in dataset[split]["triplet"]]
    tags = [item["pos_tag"] for item in dataset[split]["triplet"]]
    
    
    new_sents, new_tags = smart_split(sents, tags, MAX_SEQ_LENGTH)
    return new_sents, new_tags


def bert_tokenizer(tokenizer, sents):
    # We will use the ready split tokens
    sents_encoding = tokenizer(sents,is_split_into_words=True,
                                return_offsets_mapping=True,
                                padding="max_length",
                                truncation=True,
                                return_tensors='pt')
    pdb.set_trace()
    return sents_encoding

def smart_tags_encoder(tag2id, tags, encodings):
    # Reference: https://huggingface.co/transformers/custom_datasets.html#token-classification-with-w-nut-emerging-entities
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
    
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def main():
    """ main method """
    args = parse_arguments()
    net_start = time.time()
    root = subprocess.Popen("git rev-parse --show-toplevel", shell=True, stdout=subprocess.PIPE)
    TOP_DIR = root.stdout.read()
    TOP_DIR = TOP_DIR.strip().decode("utf-8")
    
    print('-'*20+"Loading dataset"+'-'*20)
    dataset = datasets.load_dataset(TOP_DIR+'/src/ontonotes4.py', data_files=args.ip_dir)

    # Determine maximum sequence length
    train_sents, train_tags = select_split(dataset, split="train")
    test_sents, test_tags = select_split(dataset, split="test")
    valid_sents, valid_tags = select_split(dataset, split="validation")
    print('Max sentence length:', max(len(max(train_sents, key=len)), 
                                    len(max(test_sents, key=len)), 
                                    len(max(valid_sents, key=len))))
    
    # Create encodings for tags by simple mapping
    all_tags = train_tags + test_tags + valid_tags
    unique_tags = set(tag for line in all_tags for tag in line)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    print("{:d} unique tags found".format(len(tag2id)))
    
    # Create encodings for tokens by Tokenization using the transformers Package
    print('-'*20+"Tokenizing dataset"+'-'*20)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    data_encoded = defaultdict()    
    data_encoded["train"] = bert_tokenizer(tokenizer, train_sents)
    data_encoded["test"] = bert_tokenizer(tokenizer, test_sents)
    data_encoded["validation"] = bert_tokenizer(tokenizer, valid_sents)
    
    # Aligning tokens and tags. Set the tags for the tokens we wish to ignore by setting to -100.
    tags_encoded = defaultdict()
    tags_encoded["train"] =  smart_tags_encoder(tag2id, train_tags, data_encoded["train"])
    tags_encoded["test"] =  smart_tags_encoder(tag2id, test_tags, data_encoded["test"])
    tags_encoded["validation"] =  smart_tags_encoder(tag2id, valid_tags, data_encoded["validation"])
    print("--Done--")
    
    # Generating BERT embeddings for the encoded tokens for the entire dataset
    out_dir = TOP_DIR + "/models/"
    if args.load_emb is True:
        if os.path.isfile(out_dir + "embeddings_ontonotes.pkl"):
            print('-'*20+"Pretrained embeddings found."+'-'*20)
            print("Loading...")
            infile = open(out_dir + "embeddings_ontonotes.pkl",'rb')
            sents_embeddings = pickle.load(infile)
            infile.close()
        else:
            print("Error. No embeddings found")
        
    else:
        print('-'*20+"Generating embeddings"+'-'*20)
        model = BertModel.from_pretrained("bert-base-cased",  output_hidden_states=True)
        model.to(DEVICE)
        model.eval()

        sents_embeddings = defaultdict()

        with torch.no_grad():
            for split in data_encoded:
                encoded_sents = data_encoded[split]["input_ids"]
                encoded_tags = tags_encoded[split]
                # Processing sequence batchwise to avoid memory error.
                batched_enc_sents = load_text_batchwise(encoded_sents, args)
                batched_enc_tags = list(load_text_batchwise(encoded_tags, args))
                for i, batch in enumerate(batched_enc_sents):
                    sents_embeddings = defaultdict()
                    outputs = model(torch.LongTensor(batch).to(DEVICE), output_hidden_states=True)
                    # pdb.set_trace()
                    ## Reference:
                    ## https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
                    ## https://medium.com/analytics-vidhya/bert-word-embeddings-deep-dive-32f6214f02bf
                    ## sum of all hidden states or we can do mean.
                    # embed = torch.stack(outputs.hidden_states, dim=0).sum(0)

                    # https://github.com/hanxiao/bert-as-service#q-bert-has-1224-layers-so-which-layer-are-you-talking-about
                    # The second-to-last layer is what Han settled on as a reasonable sweet-spot.
                    
                    # use last hidden state as word embeddings
                    embed = outputs.hidden_states[0]
        
                    # sents_embeddings.append({
                    #     'sent':batch,
                    #     "embedding": embed
                    # })
                    # pdb.set_trace()
                    try:
                        sents_embeddings["labels"] = batched_enc_tags[i]
                    except IndexError:
                        pdb.set_trace()
                    sents_embeddings["embeddings"] = embed
                    del outputs
                    os.makedirs(out_dir, exist_ok=True)
                    f_name = out_dir + "embeddings_" + split + "_batch_id_" + str(i) + ".pkl"
                    outfile = open(f_name,'wb')
                    pickle.dump(sents_embeddings, outfile)
                    outfile.close()
                    print("Embeddings saved at "+str(f_name))
        print("{:d} batches processed".format(len(sents_embeddings)))
        
        # print(sents_embeddings[0]['sent'].size())
        # First dim: batch size (default 64), Second dim: number of encodings in a seq
        # third dim: number of features for one encoding.
        # print(sents_embeddings[0]['embedding'].size())            
        # Dump embeddings
        if args.save_emb:
            # os.makedirs(out_dir, exist_ok=True)
            # outfile = open(out_dir+"embeddings_ontonotes.pkl",'wb')
            # pickle.dump(sents_embeddings, outfile)
            # outfile.close()
            # print("Embeddings saved at "+str(out_dir+"embeddings_ontonotes.pkl"))
            
            # Save encoded sequences, encoded tags and unique tags as pkl
            outfile = open(out_dir+"data_encoded.pkl",'wb')
            pickle.dump(data_encoded, outfile)
            outfile.close()
            print("Embeddings saved at "+str(out_dir+"data_encoded.pkl"))
            
            outfile = open(out_dir+"tags_encoded.pkl",'wb')
            pickle.dump(tags_encoded, outfile)
            outfile.close()
            print("Embeddings saved at "+str(out_dir+"tags_encoded.pkl"))
            
            outfile = open(out_dir+"tag2id.pkl",'wb')
            pickle.dump(tag2id, outfile)
            outfile.close()
            print("Embeddings saved at "+str(out_dir+"tag2id.pkl"))
            
    net_end = time.time()
    print("Total runtime: %.3f s" % (net_end - net_start))
    print("--Done--")

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ip_dir", help="Ontonotes4 dir with conll gold files")
    parser.add_argument("-save_emb", default=True, type=bool, help='save BERT embeddings')
    parser.add_argument("-load_emb", default=False, type=bool, help='choice to load pre-trained embeddings')
    parser.add_argument("-batch_size", default=64, type=int, help='batch size when generating embeddings')
    
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()
    
