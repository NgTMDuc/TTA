"""
Copyright to DeYO Authors, ICLR 2024 Spotlight (top-5% of the submissions)
built upon on Tent code.
"""

from copy import deepcopy
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F
import copy
class DeYO_Custom(nn.Module):
    """DeYO online adapts a model by entropy minimization with entropy and PLPD filtering & reweighting during testing.
    Once DeYOed, a model adapts itself by updating on every forward.
    """
    def __init__(self,
                 model, 
                 args, 
                 optimizer, 
                 steps=1, 
                 episodic=False, 
                 deyo_margin=0.5*math.log(1000), 
                 margin_e0=0.4*math.log(1000),
                 ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.args = args
        if args.wandb_log:
            import wandb
        self.steps = steps
        self.episodic = episodic
        args.counts = [1e-6,1e-6,1e-6,1e-6]
        args.correct_counts = [0,0,0,0]

        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0
        self.anchors, self.pseudo_labels = self.generate_anchor()
        self.device = next(self.model.parameters()).device
        self.embedding = self.get_embedding_layer()
        self.eps = args.alpha_cap
        self.num_sample = args.num_sim
        # for name, param in self.model.named_parameters():
            # print(f"{name}: requires_grad={param.requires_grad}")
    def generate_anchor(self):
        anchors = []
        num_classes = self.model.fc.out_features 
        pseudo_labels = []
        for class_idx in range(num_classes):
            target_vector = torch.zeros((1, num_classes)).to(next(self.model.parameters()).device)
            target_vector[0, class_idx] = 1.0
            pseudo_labels.append(target_vector)
            
            input_embedding = torch.randn((1, self.model.fc.in_features)).to(next(self.model.parameters()).device)
            input_embedding.requires_grad = True

            optimizer = torch.optim.Adam([input_embedding], lr=0.001)

            for _ in range(200):
                optimizer.zero_grad()

                output = self.model.fc(input_embedding)

                loss = -torch.nn.functional.log_softmax(output, dim=1)[0, class_idx]
                
                loss.backward()
                optimizer.step()

            anchors.append(input_embedding.detach().clone())

        return anchors, pseudo_labels
    def get_embedding(self, x):
        return self.embedding(x)
    
    def compute_ulb_grads(self, ulb_embed, pseudo_label):
        var_emb = Variable(ulb_embed, requires_grad=True).to(self.device)
        output = self.model.fc(var_emb)
        loss =  F.cross_entropy(output, pseudo_label)
        # print(loss.requires_grad)
        grads = torch.autograd.grad(loss, var_emb)[0].data
        del loss, var_emb, output
        return grads
    
    def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
        z = (lb_embedding - ulb_embedding) #* ulb_grads
        alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)
        return alpha

    
    def mix_feature(self, ulb_embed, anchor, alpha):
        return (1 - alpha) * ulb_embed + alpha * anchor
    
    def filter_sample(self, x, eps, anchors , num_sample):
        B = x.shape[0]
        num_classes = self.model.fc.out_features
        labels_count = torch.zeros((B,num_classes))
        if self.args.model == "vitbase_timm":
            ulbs_embed = self.embedding(x)[:, 0, :]
        else:
            ulbs_embed = self.embedding(x)
        predictions = torch.argmax(self.model(x).detach().cpu(), dim = 1)
        for i in tqdm.tqdm(range(B)):
            labels_count[i][predictions[i]] += 1
            for j in range(num_classes):
                anchor = anchors[j]
                label = self.pseudo_labels[j]
                ulb_embed = ulbs_embed[i].reshape(1, self.model.fc.in_features)
                grad = self.compute_ulb_grads(ulb_embed, label)
                alpha = self.calculate_optimum_alpha(eps, anchor, ulb_embed, grad)
                feature_mix = self.mix_feature(ulb_embed, anchor, alpha)
                pred = torch.argmax(self.model.fc(feature_mix))
                labels_count[i][pred] += 1
        below_threshold = labels_count < num_sample
        all_below_threshold = torch.all(below_threshold, dim=1)
        return all_below_threshold
    
    def get_filter(self, x):
        filter_ids_0 = []
        for tmp in x:
            filter_ids_0.append(self.filter_sample(tmp, self.eps, self.anchors, self.num_sample))
        return filter_ids_0
    
    
    def get_embedding_layer(self):
        return copy.deepcopy(torch.nn.Sequential(*(list(self.model.children())[:-1])))
    
    def forward(self, x, iter_, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()
        
        
        if targets is None:
            for _ in range(self.steps):
                filter_ids_0 = torch.where((self.filter_sample(x, self.eps, self.anchors, self.num_sample)))
                if flag:
                    outputs, backward, final_backward = forward_and_adapt_deyo(x, iter_, self.model, self.args,
                                                                              self.optimizer, self.deyo_margin,
                                                                              self.margin_e0, targets, flag, group, filter_ids_0)
                else:
                    outputs = forward_and_adapt_deyo(x, iter_, self.model, self.args,
                                                    self.optimizer, self.deyo_margin,
                                                    self.margin_e0,  targets, flag, group, filter_ids_0)
        else:
            for _ in range(self.steps):
                filter_ids_0 = torch.where((self.filter_sample(x, self.eps, self.anchors, self.num_sample)))
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_deyo(x, iter_, self.model, 
                                                                                                    self.args, 
                                                                                                    self.optimizer, 
                                                                                                    self.deyo_margin,
                                                                                                    self.margin_e0,
                                                                                                    targets, flag, group, filter_ids_0)
                else:
                    outputs = forward_and_adapt_deyo(x, iter_, self.model, 
                                                    self.args, self.optimizer, 
                                                    self.deyo_margin,
                                                    self.margin_e0,
                                                    targets, flag, group, filter_ids_0)
        if targets is None:
            if flag:
                return outputs, backward, final_backward
            else:
                return outputs
        else:
            if flag:
                return outputs, backward, final_backward, corr_pl_1, corr_pl_2
            else:
                return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_deyo( x, 
                            iter_, 
                            model, 
                            args, 
                            optimizer, 
                            deyo_margin, 
                            margin, 
                            targets=None, 
                            flag=True, 
                            group=None,
                            filter_ids_0 = None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    if not flag:
        return outputs
    
    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    # filter_ids_0 = torch.where((self.filter_sample(x, self.eps, self.anchors, self.num_sample)))
    if filter_ids_0 is not None:
        
        entropys = entropys[filter_ids_0]
    
        backward = len(entropys)
    # print(backward)
        if backward == 0:
            if targets is not None:
                return outputs, 0, 0 , 0 ,0
            return outputs, 0, 0
        
    # print("Done running filter 0")
        
    if args.filter_ent:
        filter_ids_1 = torch.where((entropys < deyo_margin))
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
        
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward==0:
        # print("Stop here")
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, 0, 0
    # print("Done running filter 1")
    x_prime = x[filter_ids_0][filter_ids_1]
    x_prime = x_prime.detach()
    # print(x_prime.shape)
    if args.aug_type=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
    elif args.aug_type=='patch':
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
        x_prime = resize_o(x_prime)
    elif args.aug_type=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
        
    with torch.no_grad():
        outputs_prime = model(x_prime)
    
    prob_outputs = outputs[filter_ids_0][filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
        
    if args.filter_plpd:
        filter_ids_2 = torch.where(plpd > args.plpd_threshold)
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)
        
    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
            
    if final_backward==0:
        del x_prime
        del plpd
            
        if targets is not None:
            return outputs, backward, 0, corr_pl_1, 0
        return outputs, backward, 0
    # print("Done running filter 2")
    plpd = plpd[filter_ids_2]
        
    if targets is not None:
        corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

    if args.reweight_ent or args.reweight_plpd:
        coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                args.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    del x_prime
    del plpd
        
    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    # print("Running")
    return outputs, backward, final_backward

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because DeYO optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what DeYO updates
    model.requires_grad_(False)
    # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

