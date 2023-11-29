import os
import sys
import cv2
import datetime
import torch
import torch.nn as nn
import numpy as np
import imageio
import yaml

SAVE_DIR = "./data/models"
SAVE_VID_DIR = "./data/videos"
YAML_DIR = "./data/config"

def save_model(model, config, name=""):
    date = datetime.datetime.now()
    map_str = f"{config['map_size'][0]}_{config['map_size'][1]}"  
    pothole_str = f"_potholes-{config['potholes']}_"  
    save_path = os.path.join(SAVE_DIR, name + pothole_str + map_str + ".pt")
    torch.save(model.state_dict(), save_path)
    print(f"SAVE MODEL AT -> {save_path}")

def save_frames(buffer, name=""):
    date = datetime.datetime.now()
    save_path = os.path.join(SAVE_VID_DIR, name + str(date) + ".gif")
    imageio.mimsave(save_path, buffer)

def print_action(action):
    action_map = {2: "UP", 3 : "DOWN", 0 : "LEFT", 1 : "RIGHT"} 
    print(f"Action : [{action_map[action]}]")

def load_model(path, model):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.net.load_state_dict(state_dict)

def load_yaml(path):
    yaml_path = os.path.join(YAML_DIR, path)
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

