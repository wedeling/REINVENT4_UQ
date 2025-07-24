"""
Retraining of the REINVENT de novo prior
----------------------------------------

Reivent 2017 article:
    
    Olivecrona, M., Blaschke, T., Engkvist, O., & Chen, H. (2017).
    Molecular de-novo design through deep reinforcement learning. 
    Journal of cheminformatics, 9(1), 48.

Some  RNN training references

https://github.com/rantsandruse/pytorch_lstm_02minibatch
https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random

# to create a pytorch dataset
from torch.utils.data import Dataset

#waitbar
from tqdm import tqdm

from reinvent.runmodes.create_model.create_reinvent import create_model
from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer

class SmilesDataset(Dataset):
    """
    Pytorch Dataset for Chembl smiles
    """

    def __init__(self, input_smiles_path):
        """
        Create a SmilesDataset object. Each smiles string will be 
        encoded as an integer sequence.

        Parameters
        ----------
        input_smiles_path : string
            The path to the file containing the smiles strings. This
            file should contain no headers and 1 smiles string per
            line.

        Returns
        -------
        None.

        """

        self.smiles = pd.read_csv(input_smiles_path).values

        data = []
        print('Converting smiles to integer sequences...')
        for smile in tqdm(self.smiles):
            tokens_i = [tokenizer.tokenize(data = smile[0])]
            idx_i = [model.vocabulary[token] for token in tokens_i[0]]
            data.append(torch.tensor(idx_i).to(device))

        self.data = data

    def __len__(self):
        """
        Get the data size.

        Returns
        -------
        int
            Number of (x, y) pairs.

        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get 1 (x, y) training sample.

        If 0,1,2,3,4,5,6,7 is the sequence:

            x = 0,1,2,3,4,5,6
            y = 1,2,3,4,5,6,7

        Parameters
        ----------
        idx : int
            An index to the dataset.

        Returns
        -------
        x : list
            The input sequence.
        y : list
            The shifted output sequence containing the next tokens.

        """
        
        if type(idx) is int:
            idx = [idx]
        
        x = [self.data[i][0:-1] for i in idx]
        y = [self.data[i][1:] for i in idx]

        return x, y

# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")

# standard hyper parameter settings

# number of layers in rnn
num_layers = 3
# number of neurons per layer
layer_size = 256
# dropout probability
dropout = 0.0
# max output sequence length
max_sequence_length = 256
# type of rnn lstm or gru
cell_type = 'lstm'
# size of embedding layer
embedding_layer_size = 256
# use layer normalization
layer_normalization = False
# use rdkit to standardize the smiles in the training data file (normally true, but gives error)
standardize = False
# preprocessed training data file (all data - calibration data)
input_smiles_path = './data/train_smiles.smi'
# preprocessed data file
# input_smiles_path = './data/all_data.smi'

# output pickle file of the network
output_model_path = './priors/retrained.prior'

# create a reinvent model
model = create_model(num_layers = num_layers, layer_size = layer_size, dropout = dropout,
                     max_sequence_length = max_sequence_length, 
                     cell_type = cell_type, embedding_layer_size = embedding_layer_size,
                     layer_normalization = layer_normalization,
                     standardize = standardize,
                     input_smiles_path = input_smiles_path, output_model_path=output_model_path,
                     metadata={})

# move rnn to device, cpu or gpu
model.network.to(device)

# tokens of the smiles strings
tokens = model.vocabulary.tokens()
n_tokens = len(tokens)
tokenizer = SMILESTokenizer()

# # split training / calibration data
# all_data = SmilesDataset(input_smiles_path)
# train_set, calibration_set = torch.utils.data.random_split(all_data, [.85, .15], 
#                                                            generator=torch.Generator(device=device).manual_seed(2024))
# print(f"train examples = {len(train_set)}, validation examples = {len(calibration_set)}")
# train_set = train_set.dataset 

# load training data directly from file
train_set = SmilesDataset(input_smiles_path)

# number of epochs
n_epoch = 5

# values from 2017 article
batch_size = 128
learning_rate = 0.001

# set rnn to training mode
model.network.train()

# SGD method
optimizer = torch.optim.Adam(model.network.parameters(), lr=learning_rate)
# the original reinvent 2017 article mentions a 0.02 learning rate decay
# this seems like a lot (maybe they mean gamma = 0.98?), will not use it for now.
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.02)

# negative log-likelihood loss function
criterion = nn.NLLLoss()

current_loss = 0
all_losses = []

for n in range(n_epoch):
    # clear the gradient as each epoch
    model.network.zero_grad()

    # create batches containing random indices of the training set
    batches = list(range(len(train_set)))
    random.shuffle(batches)
    batches = np.array_split(batches, len(batches) // batch_size)

    j = 0
    for batch in tqdm(batches):
        # (x, y) training pairs
        x, y = train_set[batch]

        # Loop over every sequence pair (x, y) in the mini batch,
        # model(x) will also work in one go, but then the sequences
        # in x will need to be padded with zeros. These zeros will mess
        # with the gradients, which can be circumvented by packing:
        # https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
        # This will require the modification of the forward subroutine though.
        batch_loss = 0
        for i in range(len(x)):
            # the output of the linear layer at the end of the rnn
            logits, _ = model.network(x[i].unsqueeze(0))
            # remove first dimension, making an 2D tensor
            logits = logits.squeeze(0)
            # compute the log_probs at every time stamp
            log_probs = logits.log_softmax(dim=1)

            loss = criterion(log_probs, y[i])
            batch_loss += loss

        # optimize parameters
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.network.parameters(), 3)
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()

        if j % 100 == 0:
            # lr = optimizer.param_groups[0]['lr']
            print(f'\n Current batch loss = {batch_loss.item()}')
            # print(f'\n Learning rate = {lr}')
        current_loss += batch_loss.item() / len(batch)
        j += 1

    all_losses.append(current_loss / len(batches) )
    print(f"{n} ({n / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
    current_loss = 0

# save the network
model.save(output_model_path)