import os
import sys
import datetime
import torch
import torch.nn as nn

SAVE_DIR = "../data/models"

def save_model(model):
    date = datetime.datetime.now()
    save_path = os.path.join(SAVE_DIR, "DQN-" + str(date) + ".pt")
    torch.save(model.state_dict(), save_path)
    print(f"SAVE MODEL AT -> {save_path}")