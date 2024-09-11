import torch
import clip
from torch.nn import Module
from torch import nn
class CustomCLIP(Module):
    def __init__(self, args, device, num_class):
        self.args = args 
        self.device = device
    
        model, _ = clip.load(self.args.model_name, device = device)
        self.embedding = model.visual
        self.output_dim = self.embedding.output_dim
        self.num_classes = num_class
        self.clf = nn.Linear(self.output_dim, num_class)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.clf(x)
        
        return x

