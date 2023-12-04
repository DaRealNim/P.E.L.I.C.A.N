import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optparse import OptionParser
from pefile import PE, PEFormatError
import glob
from colorama import Fore, Back, Style

from model import ConvMalware


def get_options():
    parser = OptionParser()
    parser.add_option("-m", "--model-path", dest="model_path", type="string",
                        help="model path")
    parser.add_option("-i", "--input-path", dest="input_path", type="string",
                        help="path to the executable to be classified")
    parser.add_option("-r", "--recursive", dest="recursive", action="store_true",
                        help="recursively classify all files in the directory")

    (options, args) = parser.parse_args()
    optdict = vars(options)
    return optdict

def PE_find_text_section(data):
    pe = PE(data=data)
    for section in pe.sections:
        if b'.text' in section.Name:
            return section.get_data()
    return None

if __name__ == '__main__':

    print("+-------------------------+")
    print("|      P.E.L.I.C.A.N      |")
    print("+-------------------------+")
    print()

    optdict = get_options()
    model_path = optdict['model_path']
    input_path = optdict['input_path']

    if model_path is None or input_path is None:
        print('Please specify model path and input path')
        exit(1)

    filenames = []
    # load input
    if "*" in input_path:
        input_path = glob.glob(input_path)
        if len(input_path) == 0:
            print("No file found")
            exit(1)
    else:
        input_path = [input_path]

    batch = []
    for path in input_path:
        if os.path.isdir(path):
            # if we are in recursive mode, we don't have subfiles in input_path,
            # add them to input_path
            if optdict['recursive']:
                # extend input_path with all files in the directory
                input_path.extend(glob.glob(os.path.join(path, "*")))
            else:
                print(f'{path} is a directory (use -r to for recursive mode)')
            continue

        with open(path, 'rb') as f:
            raw = f.read()

        try:
            text_section = PE_find_text_section(raw)
        except PEFormatError:
            print(f'Invalid PE file: {path}')
            continue

        if text_section is None:
            print(f'No text section found in {path}')
            continue
        batch.append(text_section)
        filenames.append(os.path.basename(path))

    if len(batch) == 0:
        print("No file found")
        exit(1)

    print(Back.WHITE + Fore.BLACK)
    sentence = f"  Let's go! {len(batch)} file(s) to process  "
    print(" " * len(sentence))
    print(sentence)
    print(" " * len(sentence))
    print(Style.RESET_ALL)

    # preprocess
    batch = [list(x) for x in batch]
    batch = [torch.tensor(x).unsqueeze(0) for x in batch]

    # load model
    model = ConvMalware()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # predict
    with torch.no_grad():
        for i, x in enumerate(batch):
            y = model(x)
            y = torch.sigmoid(y).item()
            malware = y > 0.5
            conf = (y if malware else 1-y) * 100
            print(filenames[i])
            print("  ->",
                  Fore.RED + "MALWARE" if malware else Fore.GREEN + "benign",
                  f"({conf:.2f}% confidence)"
            )
            print(Style.RESET_ALL)
