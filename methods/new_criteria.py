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
        for class_idx in tqdm.tqdm(range(num_classes)):
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
            self.log_anchor(target_vector, anchors[class_idx])
        return anchors, pseudo_labels
    
    def get_embedding(self, x):
        return self.embedding(x)
    
    def compute_ulb_grads(self, ulb_embed_batch, pseudo_label_batch):
        var_emb = ulb_embed_batch.requires_grad_(True).to(self.device)
        output = self.model.module.fc(var_emb) 
        loss = F.cross_entropy(output, pseudo_label_batch, reduction='mean')  
        grads = torch.autograd.grad(loss, var_emb, retain_graph=True)[0]  
        del loss, var_emb, output
        return grads

    def calculate_optimum_alpha(self, eps, lb_embedding_batch, ulb_embedding_batch, ulb_grads_batch):
        z = lb_embedding_batch - ulb_embedding_batch  # Shape: [batch_size, embedding_dim]
        z_norm = z.norm(dim=1, keepdim=True)  # Shape: [batch_size, 1]
        ulb_grads_norm = ulb_grads_batch.norm(dim=1, keepdim=True)  # Shape: [batch_size, 1]
        alpha = (eps * z_norm / (ulb_grads_norm + 1e-8)).repeat(1, z.size(1)) * ulb_grads_batch / (z + 1e-8)

        return alpha
    
    def mix_feature(self, ulb_embed_batch, anchor_batch, alpha_batch):
        # ulb_embed_batch: tensor of shape [batch_size, embedding_dim]
        # anchor_batch: tensor of shape [batch_size, embedding_dim]
        # alpha_batch: tensor of shape [batch_size, embedding_dim]
        
        # Mixing the features using batch-wise tensor operations
        mixed_features = (1 - alpha_batch) * ulb_embed_batch + alpha_batch * anchor_batch
        # Shape of mixed_features: [batch_size, embedding_dim]
        
        return mixed_features
        
    def filter_sample(self, x, eps, anchors, num_sample):
        # x: tensor of shape [batch_size, channels, height, width]
        # eps: perturbation parameter
        # anchors: list of anchor embeddings, each of shape [1, embedding_dim]
        # num_sample: number of simulations (threshold)

        B = x.shape[0]  # Batch size
        num_classes = self.model.module.fc.out_features
        labels_count = torch.zeros((B, num_classes))

        # Get embeddings from the model's embedding layer
        if self.args.model == "vitbase_timm":
            ulbs_embed = self.embedding(x)[:, 0, :]  # Shape: [batch_size, embedding_dim]
        else:
            ulbs_embed = self.embedding(x)  # Shape: [batch_size, embedding_dim]

        # Get the predictions for the original batch of samples
        predictions = torch.argmax(self.model(x).detach().cpu(), dim=1)
        
        # Update label counts for the original predictions
        for i in range(B):
            labels_count[i][predictions[i]] += 1

        # Iterate over each class to compute gradients and perform feature mixing for the whole batch
        for j in range(num_classes):
            # Anchor and pseudo label for the current class
            anchor = anchors[j].repeat(B, 1)  # Shape: [batch_size, embedding_dim]
            label = self.pseudo_labels[j].repeat(B, 1)  # Shape: [batch_size, num_classes]
            # print("Check at line 110")
            # print(ulbs_embed.shape)
            # print(label.shape)
            ulbs_embed = ulbs_embed.view(ulbs_embed.size(0), -1)

            # Compute gradients for the whole batch
            grads = self.compute_ulb_grads(ulbs_embed, label)  # Shape: [batch_size, embedding_dim]

            # Calculate the optimal alpha for each sample in the batch
            alpha = self.calculate_optimum_alpha(eps, anchor, ulbs_embed, grads)  # Shape: [batch_size, embedding_dim]

            # Mix features for each sample in the batch
            feature_mix = self.mix_feature(ulbs_embed, anchor, alpha)  # Shape: [batch_size, embedding_dim]

            # Compute predictions for the mixed features
            mixed_output = self.model.module.fc(feature_mix)  # Shape: [batch_size, num_classes]
            pred = torch.argmax(mixed_output, dim=1)  # Shape: [batch_size]

            # Update label counts for each sample in the batch
            for i in range(B):
                labels_count[i][pred[i]] += 1

        # Determine which samples are below the threshold for number of simulations
        below_threshold = labels_count < num_sample
        all_below_threshold = torch.all(below_threshold, dim=1)  # Shape: [batch_size]

        return all_below_threshold

    
    def log_anchor(self, label, anchor):
        with open(self.args.save_path, "a") as f:
            f.write("Label: \n")
            f.write(str(label))
            f.write("Anchor: \n")
            f.write(str(anchor))