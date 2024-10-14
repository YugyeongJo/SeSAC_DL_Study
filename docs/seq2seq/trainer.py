import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

from torch.utils.data import DataLoader

class Trainer:
    """_summary_
    """
    
    def __init__(self, model, device, criterion, optimizer):
        """_summary_

        Args:
            model (_type_): _description_
            device (_type_): _description_
            criterion (_type_): _description_
            optimizer (_type_): _description_
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        
    def train(self, ):
        """_summary_
        """
        