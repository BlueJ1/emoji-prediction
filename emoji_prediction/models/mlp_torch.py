import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.evaluate_predictions import evaluate_predictions


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
             ("bn1", nn.BatchNorm1d(512)),
             ("relu1", nn.ReLU()),
             ("fc2", nn.Linear(512, 256)),
             ("bn2", nn.BatchNorm1d(256)),
             ("relu2", nn.ReLU()),
             ("fc3", nn.Linear(256, 128)),
             ("bn3", nn.BatchNorm1d(128)),
             ("relu3", nn.ReLU()),
             ("fc4", nn.Linear(128, 128)),
             ("bn4", nn.BatchNorm1d(128)),
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
        f1 = f1_score(y_true.detach().cpu().numpy(),
                      np.argmax(y_pred.detach().cpu().numpy(), axis=1), average="weighted")
        return loss, f1


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
    # optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model.to(device)

    # Standardize the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()

    train_dataset = EmojiDataset(X_train, y_train)
    test_dataset = EmojiDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    with tqdm(total=num_epochs, unit="epoch") as progress_bar:
        for epoch in range(num_epochs):
            model.train()
            losses = []
            f1_scores = []
            for i, (X_batch, y_batch) in enumerate(train_dataloader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss, train_f1 = model.loss(y_pred, y_batch)

                losses.append(loss.item())
                f1_scores.append(train_f1)

                model.zero_grad()
                loss.backward()
                optimizer.step()

            val_losses = []
            val_f1_scores = []

            model.eval()
            for i, (X_batch, y_batch) in enumerate(test_dataloader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss, test_f1 = model.loss(y_pred, y_batch)

                val_losses.append(loss.item())
                val_f1_scores.append(test_f1)

            progress_bar.set_postfix_str(f"Loss: {np.mean(losses):.2f}, F1 Score: {np.mean(f1_scores) * 100:.2f}%"
                                         f"Val Loss: {np.mean(val_losses):.2f}, Val F1 Score: {np.mean(val_f1_scores) * 100:.2f}%")
            progress_bar.update(1)

    model.eval()


    evaluation = evaluate_predictions(y_pred, y_test)
    print(evaluation["weighted_f1_score"])
    results_dict[fold_number] = evaluation


def mlp_data(df):
    X = np.stack(df["words"].values)
    # expanded_df = df.apply(lambda row: pd.Series(row['words']), axis=1)

    # Concatenate the expanded columns with the original DataFrame
    # result_df = pd.concat([df, expanded_df], axis=1)

    # Drop the original array_column
    # result_df = result_df.drop('words', axis=1)

    # X = result_df.iloc[:, 49:].values
    y = np.argmax(df.iloc[:, 1:].values, axis=1)

    print(X.shape, y.shape)
    # print(X)
    # print(y)
    return X, y
