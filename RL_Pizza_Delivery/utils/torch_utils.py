import os
import sys
import cv2
import datetime
import torch
import torch.nn as nn
import numpy as np
import imageio

SAVE_DIR = "../data/models"
SAVE_VID_DIR = "../data/videos"

def save_model(model):
    date = datetime.datetime.now()
    save_path = os.path.join(SAVE_DIR, "DQN-" + str(date) + ".pt")
    torch.save(model.state_dict(), save_path)
    print(f"SAVE MODEL AT -> {save_path}")

def save_frames(buffer):
    date = datetime.datetime.now()
    save_path = os.path.join(SAVE_VID_DIR, "DQN-VID-" + str(date) + ".gif")
    imageio.mimsave(save_path, buffer)

