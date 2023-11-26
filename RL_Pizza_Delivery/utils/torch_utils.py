import os
import sys
import cv2
import datetime
import torch
import torch.nn as nn
import numpy as np

SAVE_DIR = "../data/models"
SAVE_VID_DIR = "../data/vids"

def save_model(model):
    date = datetime.datetime.now()
    save_path = os.path.join(SAVE_DIR, "DQN-" + str(date) + ".pt")
    torch.save(model.state_dict(), save_path)
    print(f"SAVE MODEL AT -> {save_path}")

def save_frames(buffer):
    date = datetime.datetime.now()
    save_path = os.path.join(SAVE_VID_DIR, "DQN-VID-" + str(date) + ".mp4")
    size = 512, 512
    fps = 25
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for img in np.array(buffer, dtype='uint8'):
        img = np.moveaxis(img, -1, 0)
        out.write(img)
    out.release()

