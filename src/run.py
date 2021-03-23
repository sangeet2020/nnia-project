

import gc
import os
import time
import torch
import wandb
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
from gru import GRUNet
from lstm import LSTMTagger
import torch.optim as optim
import torch.nn.functional as F


def find(name, path):
    paths = []
    for root, dirs, files in os.walk(path):
        for efile in files:
            if name in efile:
                paths.append(os.path.join(root, efile))
    return paths

def get_batches(train_paths):
    
    datasets = []
    for i in range(0, len(train_paths),2):
        train_path = train_paths[i:i+2]
        file = open(train_path[0],'rb')
        dataset1 = pickle.load(file)
        file.close()
        dataset1["labels"] = torch.tensor(dataset1["labels"], dtype=int).narrow(1,0,512)
        if len(train_path) == 2:
            file = open(train_path[1],'rb')
            dataset = pickle.load(file)
            file.close()
            dataset["labels"] = torch.tensor(dataset["labels"], dtype=int).narrow(1,0,512)
            dataset1["embeddings"] = torch.cat((dataset1["embeddings"], dataset["embeddings"]), 0)
            dataset1["labels"] = torch.cat((dataset1["labels"], dataset["labels"]), 0)
        else:
            dataset1["embeddings"] = dataset1["embeddings"].narrow(1,0,512)
        yield dataset1
        
        
def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = False) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx)
    correct = max_preds[non_pad_elements].eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)



def train(model, train_paths, val_paths, args):
    
    optimizer = optim.Adam(model.parameters(), lr = 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index = -100)
    
    for i in range(args.epochs):
        count  = 0
        epoch_loss = 0
        epoch_acc = 0

        train_batches = get_batches(train_paths)
        model.train()
        for each_batch in train_batches:
            x = each_batch["embeddings"]
            y = each_batch["labels"]
            optimizer.zero_grad()
            predictions = model(x).permute(0,2,1)
            loss = criterion(predictions, y)
            acc = categorical_accuracy(predictions, y, -100)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            count += 1

        loss = epoch_loss/count
        accuracy = epoch_acc/count
        
        print("Epoch: {:d} Train loss: {:.2f}, Train acc: {:.2f}".format(i, loss, accuracy))
        wandb.log({"train_loss": loss})
        wandb.log({"train_acc": accuracy})

        del train_batches
        gc.collect()
        val_batches = get_batches(val_paths)
        val_loss = []
        val_acc = []
        for each_batch in val_batches:

            x = each_batch["embeddings"]
            y = each_batch["labels"]
            predictions = model(x).permute(0,2,1)
            loss = criterion(predictions, y)
            acc = categorical_accuracy(predictions, y, -100)
            val_loss.append(loss.item())
            val_acc.append(acc.item())

        loss = sum(val_loss)/len(val_loss)
        accuracy = sum(val_acc)/len(val_acc)
        wandb.log({"val_loss": loss})
        wandb.log({"val_acc": accuracy})
        
        # print("Epoch: {:d} Loss: {:.2f}, Accuracy: {:.2f}".format(i, loss, accuracy))
    print("***** Done ******")
    return model

        
def test(model, test_paths, args):
    criterion = nn.CrossEntropyLoss(ignore_index = -100)
    test_loss = []
    test_acc = []
    test_batches = get_batches(test_paths)
    for each_batch in test_batches:

        x = each_batch["embeddings"]
        y = each_batch["labels"]
        predictions = model(x).permute(0,2,1)
        loss = criterion(predictions, y)
        acc = categorical_accuracy(predictions, y, -100)
        test_loss.append(loss.item())
        test_acc.append(acc.item())

    loss = sum(test_loss)/len(test_loss)
    accuracy = sum(test_acc)/len(test_acc)

    print("Loss: {:.2f}\nAccuracy: {:.2f} ".format(loss, accuracy))

def main():
    args = parse_arguments()

    
    train_paths = find("train", args.emb_dir)
    val_paths = find("validation", args.emb_dir)
    test_paths = find("test", args.emb_dir)
    
    tagsid_path = args.emb_dir + "/tag2id.pkl"
    file = open(tagsid_path,'rb')
    tagstoid = pickle.load(file)
    file.close()
    unique_tags = tagstoid.keys()
    
    EMBEDDING_DIM = 768
    HIDDEN_DIM = args.hidden_dim
    OUTPUT_DIM = len(unique_tags)
    N_LAYERS = args.num_layers
    DROPOUT = args.dropout
    BATCH_SIZE = args.batch_size
    if args.model_choice == "lstm":
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,OUTPUT_DIM, DROPOUT)
    elif args.model_choice == "gru":
        model = GRUNet(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS)
    else:
        print("Error. Model not found. Try again.")
        exit()
        
    optimizer = optim.Adam(model.parameters(), lr = 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index = -100)
    wandb.init(project="LSTM Bert POS tagging")
    wandb.watch(model)
    trnd_model = train(model, train_paths, val_paths, args)
    test(trnd_model, test_paths, args)
    
    
        
def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_choice", help="choose your model: lstm, gru")
    parser.add_argument("emb_dir", help="path to embeddings dir")
    parser.add_argument("-num_layers", default=2, type=int, help='number of hidden layers')
    parser.add_argument("-dropout", default=0.20, type=float, help='dropout parameter')
    parser.add_argument("-batch_size", default=64, type=int, help='batch size')
    parser.add_argument("-hidden_dim", default=256, type=int, help='dimension of hidden layers')
    parser.add_argument("-epochs", default=10, type=int, help='number of training epochs')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()