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

def print_action(action):
    action_map = {2: "UP", 3 : "DOWN", 0 : "LEFT", 1 : "RIGHT"} 
    print(f"Action : [{action_map[action]}]")

def load_model(path, model):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.net.load_state_dict(state_dict)