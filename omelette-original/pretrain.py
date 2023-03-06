import argparse
import copy
from PropLang import PropLang
from MathLang import MathLang

from pytorch_lightning.plugins import SingleDevicePlugin, DDPSpawnPlugin
from ppo import PPOAgent, get_lang_from_str
from rejoice.pretrain_dataset_gen import generate_dataset
import torch_geometric as pyg
from rejoice.PretrainingDataset import PretrainingDataset
from rejoice.tests.test_lang import TestLang
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import os
import torchmetrics
import torch
from multiprocessing import Pool
from itertools import repeat
import numpy as np
import concurrent.futures
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", type=bool, default=False,
                        help="the number of node features for input graphs")
    parser.add_argument("--count", type=int, default=100_000,
                        help="the number of expressions to generate")
    parser.add_argument("--num-threads", type=int, default=2,
                        help="the number of threads to spawn")
    parser.add_argument("--seed", type=int, default=1,
                        help="the random seed used for generation")
    parser.add_argument("--lang", type=str, default="PROP",
                        help="The language name to execute")
    args = parser.parse_args()
    return args


def split_dataset(dataset, train=0.6, val=0.3, test=0.1):
    t = int(train * len(dataset))
    v = int((train + val) * len(dataset))
    train_dataset = dataset[:t]
    val_dataset = dataset[t:v]
    test_dataset = dataset[v:]
    return train_dataset, val_dataset, test_dataset


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class PretrainModule(pl.LightningModule):
    def __init__(self, model, batch_size: int, learning_rate=1e-4):
        super().__init__()
        # self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = model
        self.accuracy = torchmetrics.Accuracy()
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data):
        x = self.model(data)
        loss = self.loss_module(input=x, target=data.y)
        acc = self.accuracy(x, data.y)
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch)
        self.log("train_loss", loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log("acc/train_acc", acc, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("acc/val_acc", acc, batch_size=self.batch_size, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch)
        self.log("acc/test_acc", acc, batch_size=self.batch_size, on_step=False, on_epoch=True)


def train(lang_name: str, seed:int):
    pl.seed_everything(seed, workers=True)
    lang = get_lang_from_str(lang_name)
    envs_mock = Struct(**{
        "single_observation_space": Struct(**{
            "num_node_features": lang.num_node_features
        }),
        "single_action_space": Struct(**{
            "n": lang.num_rules + 1
        })
    })

    batch_size = 128

    agent = PPOAgent(envs=envs_mock, use_dropout=True)
    model = PretrainModule(copy.deepcopy(agent.actor),
                           batch_size=batch_size, learning_rate=1e-4)
    dataset = PretrainingDataset(lang=lang, root=lang.name).shuffle()
    train_data, val_data, test_data = split_dataset(dataset)

    pyg_lightning_dataset = pyg.data.LightningDataset(train_dataset=train_data,
                                                      val_dataset=val_data,
                                                      test_dataset=test_data,
                                                      batch_size=batch_size,
                                                      num_workers=4
                                                      )

    trainer = pl.Trainer(strategy=DDPSpawnPlugin(find_unused_parameters=False),
                         precision=16,
                         accelerator='gpu',
                         gradient_clip_val=0.5,
                         devices=1,
                         check_val_every_n_epoch=1,
                         max_epochs=5000,
                         deterministic=True)
    trainer.fit(model, pyg_lightning_dataset)

    torch.save(model.model.state_dict(), f"./{lang.name}_weights.pt")


def gen(x):
    # seed on this process
    count, lang_name, seed = x
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    lang = get_lang_from_str(lang_name)
    generate_dataset(lang, num=count, rng=rng)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.generate == True:
        print("initializing with seed", args.seed)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            t = [(args.count, args.lang, args.seed + i) for i in range(args.count)]
            executor.map(gen, t)
    else:
        train(args.lang, args.seed)


if __name__ == "__main__":
    main()
