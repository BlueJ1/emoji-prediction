import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from keras.metrics import F1Score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluate_predictions import evaluate_predictions


class EmojiDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fcs = nn.Sequential(OrderedDict(
            [("fc1", nn.Linear(input_dim, 512)),
             ("relu1", nn.ReLU()),
             ("fc2", nn.Linear(512, 256)),
             ("relu2", nn.ReLU()),
             ("fc3", nn.Linear(256, 128)),
             ("relu3", nn.ReLU()),
             ("fc4", nn.Linear(128, 128)),
             ("relu4", nn.ReLU()),
             ("fc5", nn.Linear(128, output_dim))]))
        self.softmax = nn.Softmax(dim=1)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fcs(x)
        x = self.softmax(x)
        return x

    def loss(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)
        f1_score = F1Score(average="weighted")(y_true.detach().cpu().numpy(),
                                               np.argmax(y_pred.detach().cpu().numpy(), axis=1))
        return loss, f1_score


def train_fold(fold_number, X_train, y_train, X_test, y_test, results_dict, hyperparameters):
    input_dim = hyperparameters['input_dim']
    output_dim = 49
    learning_rate = hyperparameters['lr']
    num_epochs = hyperparameters['num_epochs']
    batch_size = hyperparameters['batch_size']

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else
    "cpu")

    # Build the MLP model
    model = MLP(input_dim, output_dim)

    model.to(device)

    # Standardize the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = EmojiDataset(X_train, y_train)
    test_dataset = EmojiDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as progress_bar:
            for i, (X_batch, y_batch) in enumerate(train_dataloader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss, f1_score = model.loss(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix_str(f"Loss: {loss:.2f}, F1 Score: {f1_score * 100:.2f}%")
                progress_bar.update(1)

    model.eval()
    y_pred = np.stack([np.argmax(model(x.to(device)).detach().cpu().numpy(), axis=1) for x, _ in test_dataloader])

    results_dict[fold_number] = evaluate_predictions(y_pred, y_test)


def mlp_data(df):
    expanded_df = df.apply(lambda row: pd.Series(row['words']), axis=1)

    # Concatenate the expanded columns with the original DataFrame
    result_df = pd.concat([df, expanded_df], axis=1)

    # Drop the original array_column
    result_df = result_df.drop('words', axis=1)

    X = result_df.iloc[:, 49:].values
    y = result_df.iloc[:, :49].values

    print(X.shape, y.shape)
    # print(X)
    # print(y)
    return X, y
