from datetime import datetime

import pandas as pd
import numpy as np
import os
# importing pytorch libraries
import torch
from torch import nn
from torch import autograd
from autoencoder import encoder, decoder

INPUT_DIM = 384

# Precompute transformation configuration from the full dataset
# This part can be run once (or at application startup)
full_dataset = pd.read_csv('data/datathon_data.csv')
categorical_attr_names = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT']
numeric_attr_names = ['DMBTR', 'WRBTR']

# Create one-hot mapping for categorical attributes based on the full dataset
full_cat_dummies = pd.get_dummies(full_dataset[categorical_attr_names])
fixed_cat_columns = full_cat_dummies.columns

# Compute numeric min and max from the full dataset (after epsilon addition)
epsilon = 1e-7
numeric = full_dataset[numeric_attr_names] + epsilon
numeric = np.log(numeric)
num_min = numeric.min()
num_max = numeric.max()


def process_data(input_data, return_label=True):
    series = False
    if isinstance(input_data, pd.Series):
        input_data = input_data.to_frame().T
        input_data = pd.concat([input_data, input_data])
        input_data.index = [input_data.index[0], input_data.index[0] + 1]
        input_data = pd.DataFrame(input_data)
        series = True
    # Remove label information as before
    if ('label' in input_data.columns) & (return_label):
        label = input_data.copy().pop('label') == 'anomal'

    # Process categorical features
    dummies = pd.get_dummies(input_data[categorical_attr_names])
    # Ensuring consistency: add missing columns with zero values, and discard extras
    dummies = dummies.reindex(columns=fixed_cat_columns, fill_value=0)

    # Process numeric features
    numeric_attr = input_data[numeric_attr_names].astype(np.float64) + epsilon
    numeric_attr = np.log(numeric_attr)
    # Normalize using fixed min and max from the full dataset
    numeric_normalized = (numeric_attr - num_min) / (num_max - num_min)
    # Merge transformed subsets
    ori_subset_transformed = pd.concat([dummies, numeric_normalized], axis=1)
    if series:
        ori_subset_transformed = pd.DataFrame(ori_subset_transformed.iloc[0]).T
    if return_label:
        return ori_subset_transformed, label
    return ori_subset_transformed


def get_recreational_error(data_transformed_to_predict):
    # restore pretrained model checkpoint
    encoder_model_name = "ep_10_encoder_model.pth"
    decoder_model_name = "ep_10_decoder_model.pth"

    # init training network classes / architectures
    encoder_eval = encoder(INPUT_DIM)
    decoder_eval = decoder(INPUT_DIM)

    # load trained models
    encoder_eval.load_state_dict(torch.load(os.path.join("models", encoder_model_name)))
    decoder_eval.load_state_dict(torch.load(os.path.join("models", decoder_model_name)))

    # set networks in evaluation mode (don't apply dropout)
    encoder_eval.eval()
    decoder_eval.eval()

    # convert encoded transactional data to torch Variable
    array = data_transformed_to_predict.values
    float_array = array.astype(np.float64)
    torch_dataset = torch.from_numpy(float_array).float()
    data = autograd.Variable(torch_dataset)

    # reconstruct encoded transactional data
    reconstruction = decoder_eval(encoder_eval(data))

    # define the optimization criterion / loss function
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')


    # init binary cross entropy errors
    reconstruction_loss_transaction_predict = np.zeros(reconstruction.size()[0])

    # iterate over all detailed reconstructions
    for i in range(0, reconstruction.size()[0]):

        # determine reconstruction loss - individual transactions
        reconstruction_loss_transaction_predict[i] = loss_function(reconstruction[i], data[i]).item()

        if (i % 100000 == 0):
            ### print conversion summary
            now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
            print('[LOG {}] collected individual reconstruction loss of: {:06}/{:06} transactions'.format(now, i,
                                                                                                          reconstruction.size()[
                                                                                                              0]))

    # Flag anomalies: True if the reconstruction loss is above the threshold
    return reconstruction_loss_transaction_predict

def get_labels(sample):
    error = get_recreational_error(sample)
    threshold = 0.00875176585942327
    return error > threshold


