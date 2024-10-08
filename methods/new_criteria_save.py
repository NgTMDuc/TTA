import pandas as pd
import torch
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import math
import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F
import copy
from torch.autograd import Variable

class Update_method(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        
        self.device = next(self.model.module.parameters()).device
        self.args = args
        self.num_sample = self.args.num_sim
        self.embedding = copy.deepcopy(torch.nn.Sequential(*(list(self.model.module.children())[:-1])))
        self.anchors, self.pseudo_labels = self.generate_anchor()
        self.eps = args.alpha_cap
        self.num_sample = args.num_sim
        
    def generate_anchor(self):
        anchors = []
        num_classes = self.model.module.fc.out_features
        pseudo_labels = []
        for class_idx in range(num_classes):
            target_vector = torch.zeros((1, num_classes)).to(self.device)
            target_vector[0, class_idx] = 1.0
            pseudo_labels.append(target_vector)
            
            input_embedding = torch.randn((1, self.model.module.fc.in_features)).to(self.device)
            input_embedding.requires_grad = True
            
            optimizer = torch.optim.Adam([input_embedding], lr=0.0001)
            
            for _ in range(self.args.epoch_anchors):
                optimizer.zero_grad()
                output = self.model.module.fc(input_embedding)
                loss = -torch.nn.functional.log_softmax(output, dim=1)[0, class_idx]
                loss.backward()
                optimizer.step()
            anchors.append(input_embedding.detach().clone())
            # self.log_anchor(target_vector, anchors[class_idx])
        return anchors, pseudo_labels
    
    def get_embedding(self, x):
        return self.embedding(x)
    
    def compute_ulb_grads(self, ulb_embed, pseudo_label):
        var_emb = ulb_embed.requires_grad_(True).to(self.device)
        output = self.model.module.fc(var_emb)
        loss = F.cross_entropy(output, pseudo_label)
        grads = torch.autograd.grad(loss, var_emb)[0]
        del loss, var_emb, output
        return grads

    def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
        z = (lb_embedding - ulb_embedding)
        alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)
        return alpha
    
    def mix_feature(self, ulb_embed, anchor, alpha):
        return (1 - alpha) * ulb_embed + alpha * anchor
    
    def filter_sample(self, x, eps, anchors, num_sample):
        B = x.shape[0]
        num_classes = self.model.module.fc.out_features
        labels_count = torch.zeros((B, num_classes))
        if self.args.model == "vitbase_timm":
            ulbs_embed = self.embedding(x)[:, 0, :]
        else:
            ulbs_embed = self.embedding(x)
        
        predictions = torch.argmax(self.model(x).detach().cpu(), dim = 1)
        for i in range(B):
            labels_count[i][predictions[i]] += 1
            for j in range(num_classes):
                anchor = anchors[j]
                label = self.pseudo_labels[j]
                ulb_embed = ulbs_embed[i].reshape(1, self.model.module.fc.in_features)
                grad = self.compute_ulb_grads(ulb_embed, label)
                alpha = self.calculate_optimum_alpha(eps, anchor, ulb_embed, grad)
                feature_mix = self.mix_feature(ulb_embed, anchor, alpha)
                pred = torch.argmax(self.model.module.fc(feature_mix))
                labels_count[i][pred] += 1
        below_threshold = labels_count < num_sample
        all_below_threshold = torch.all(below_threshold, dim=1)
        return all_below_threshold
    
    def log_anchor(self, label, anchor):
        with open(self.args.save_path, "a") as f:
            f.write("Label: \n")
            f.write(str(label))
            f.write("Anchor: \n")
            f.write(str(anchor))