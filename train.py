from optparse import OptionParser
import warnings
warnings.filterwarnings("ignore", ".*is_sparse is deprecated and will be removed.*")
warnings.filterwarnings("ignore", ".*Applied workaround for CuDNN issue.*")

import numpy as np
import pandas as pd
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model import ConvMalware

def get_options():
    parser = OptionParser()
    parser.add_option("-e", "--epochs", dest="epochs", default=10, type="int",
                      help="number of epochs")
    parser.add_option("-b", "--batch_size", dest="batch_size", default=64, type="int",
                      help="batch size")
    parser.add_option("-a", "--accumulate", dest="accumulate", default=1, type="int",
                      help="accumulate gradient every x batches")
    parser.add_option("-l", "--learning_rate", dest="learning_rate", default=0.01, type="float",
                      help="learning rate")
    parser.add_option("-d", "--datapkl", dest="datapkl", default="data.pkl", type="string",
                      help="data pickle file")
    parser.add_option("-c", "--checkpoint-path", dest="checkpoint_path", default="checkpoint", type="string",
                      help="checkpoint path")
    parser.add_option("-o", "--output-path", dest="output_path", default="output", type="string",
                      help="model output path")
    parser.add_option("--random-seed", dest="random_seed", default=42, type="int",
                      help="random seed for data splitting and model initialization")
    parser.add_option("--test-ratio", dest="test_ratio", default=0.2, type="float",
                      help="ratio of test data") 
    parser.add_option("--device", dest="device", default="cuda", type="string",
                      help="device to use (cuda or cpu)")

    # parse options into a dictionary
    (options, args) = parser.parse_args()
    optdict = vars(options)
    return optdict


def preprocess(data):
    data['label'].replace({'malware': 1, 'goodware': 0}, inplace=True)
    data = data[data["text_bytes"].apply(len) != 0]
    return data


def order_by_len(df):
    df = df.sort_values(by="text_bytes", key=lambda x: x.str.len())
    return df
    

def bytestr_batch_to_tensor(batch_X):
    batch_X = [list(x) for x in batch_X.values]

    # pad to max length
    max_len = max([len(x) for x in batch_X])
    batch_X = np.array([np.pad(x, (0, max_len - len(x)), 'constant', constant_values=256) for x in batch_X])
    batch_X = torch.tensor(batch_X, dtype=torch.int)
    return batch_X


def classification_report(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    
    print("  Accuracy\tPrecision\tRecall\tF1\tAUC")
    # cap at 2 decimal places
    print(f"  {acc:.2f}\t\t{prec:.2f}\t\t{rec:.2f}\t{f1:.2f}\t{auc:.2f}")



if __name__ == '__main__':
    optdict = get_options()

    # load data
    with open(optdict['datapkl'], 'rb') as f:
        data = pickle.load(f)

    # preprocess data
    data = preprocess(data)

    # split data
    df_train, df_test = train_test_split(
        data,
        test_size=optdict['test_ratio'],
        random_state=optdict['random_seed']
    )

    # order by length
    df_train = order_by_len(df_train)
    df_test = order_by_len(df_test)

    # split into X and y
    X_train = df_train['text_bytes']
    y_train = df_train['label']
    X_test = df_test['text_bytes']
    y_test = df_test['label']

    device = torch.device(optdict['device'])

    # model
    model = ConvMalware()
    model.to(device)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=optdict['learning_rate'])

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    batch_size = optdict['batch_size']
    accumulate = optdict['accumulate']
    n_epochs = optdict['epochs']

    for epoch in range(n_epochs):
        # train
        model.train()
        optimizer.zero_grad()
        avg_epoch_loss = 0
        
        print(f"Epoch: {epoch+1}/{n_epochs}")
        i = 0
        for batch_idx in range(0, len(X_train), batch_size):
            i += 1
            if batch_idx % 200 == 0:
                print(f"  Sample: {batch_idx}/{len(X_train)}", end="\r", flush=True)
            batch_X = X_train[batch_idx : batch_idx+batch_size]
            batch_y = y_train[batch_idx : batch_idx+batch_size]
            batch_X = bytestr_batch_to_tensor(batch_X).to(device)
            batch_y = torch.tensor(batch_y.values, dtype=torch.float).unsqueeze(1)

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_X)
            #if batch size is 1, output is a scalar: unsqueeze
            if len(output.shape) == 0:
                output = output.unsqueeze(0)

            loss = criterion(output, batch_y)
            loss.backward()

            if (i) % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_epoch_loss += loss.item()

        avg_epoch_loss /= len(X_train)
        print()
        print(f"  Loss: {avg_epoch_loss}")
        # eval
        model.eval()
        with torch.no_grad():
            test_preds = []
            test_probs = []
            print ("  Evaluating...", end="\r", flush=True)
            for i in range(0, len(X_test), optdict['batch_size']):
                batch_X = X_test[i:i+optdict['batch_size']]
                batch_y = y_test[i:i+optdict['batch_size']]
                batch_X = bytestr_batch_to_tensor(batch_X).to(device)
                batch_y = torch.tensor(batch_y.values, dtype=torch.float).unsqueeze(1)

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                output = model(batch_X)
                probs = torch.sigmoid(output)
                batchpreds = (probs > 0.5).int()
                test_preds.extend(batchpreds.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())

            print()
            classification_report(y_test, test_preds, test_probs)
            print()
            print()

        # save a checkpoint
        if not os.path.exists(optdict['checkpoint_path']):
            os.makedirs(optdict['checkpoint_path'])
        torch.save(model.state_dict(), os.path.join(optdict['checkpoint_path'], f"epoch_{epoch+1}.pt"))


