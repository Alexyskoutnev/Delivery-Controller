import os
import sys
import cv2
import datetime
import torch
import torch.nn as nn
import numpy as np
import imageio
import yaml

SAVE_DIR = "./data/final_models"
SAVE_VID_DIR = "./data/videos"
YAML_DIR = "./data/config"

def save_model(model, config, name="", type='ppo'):
    """
    Save the PyTorch model to a file.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        config (dict): Configuration parameters related to the model.
        name (str): Additional name for the saved file.
        type (str): Type of model ('dqn' for Deep Q-Network, 'ppo' for Proximal Policy Optimization).
    """
    date = datetime.datetime.now()
    map_str = f"{config['map_size'][0]}_{config['map_size'][1]}"  
    pothole_str = f"_potholes-{config['potholes']}_"  
    save_path = os.path.join(SAVE_DIR, name + pothole_str + map_str + ".pt")
    if type == 'dqn':
        torch.save(model.state_dict(), save_path)
        print(f"SAVE MODEL AT -> {save_path}")
    elif type == 'ppo':
        checkpoint = {
            'actor_state_dict': model.actor.state_dict(),
            'critic_state_dict': model.critic.state_dict()
        }
        torch.save(checkpoint, save_path)
        print(f'Model weights saved to {save_path}')

def save_frames(buffer, name=""):
    """
    Save a sequence of frames as a GIF.

    Args:
        buffer (list): List of frames to be saved.
        name (str): Additional name for the saved file.
    """
    date = datetime.datetime.now()
    save_path = os.path.join(SAVE_VID_DIR, name + str(date) + ".gif")
    imageio.mimsave(save_path, buffer)

def print_action(action):
    """
    Print the corresponding action based on its numerical code.

    Args:
        action (int): Numerical code representing the action.
    """
    action_map = {2: "UP", 3 : "DOWN", 0 : "LEFT", 1 : "RIGHT"} 
    print(f"Action : [{action_map[action]}]")

def load_model(path, model, type="ppo"):
    """
    Load weights into a PyTorch model from a file.

    Args:
        path (str): Path to the file containing model weights.
        model (torch.nn.Module): The PyTorch model to which weights will be loaded.
        type (str): Type of model ('dqn' for Deep Q-Network, 'ppo' for Proximal Policy Optimization).
    """
    if type == 'dqn':
        try:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            model.net.load_state_dict(state_dict)
            print(f'Model weights loaded from {path}')
        except FileNotFoundError:
            print(f'Error: File not found - {path}')
        except Exception as e:
            print(f'Error loading model weights: {e}')  
    elif type == 'ppo':
        try:
            checkpoint = torch.load(path)
            model.actor.load_state_dict(checkpoint['actor_state_dict'])
            model.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f'Model weights loaded from {path}')
        except FileNotFoundError:
            print(f'Error: File not found - {path}')
        except Exception as e:
            print(f'Error loading model weights: {e}')

def load_yaml(path):
    """
    Load YAML configuration from a file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Loaded YAML configuration.
    """
    yaml_path = os.path.join(YAML_DIR, path)
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    """
    Get the PyTorch device (CPU or GPU) available.

    Returns:
        torch.device: PyTorch device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

