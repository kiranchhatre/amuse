
import torch
import einops
from torch import nn
from torchmetrics import Metric
from torch.nn import functional as F

class DiffusionLosses(Metric):
    pass

class HuberLoss(nn.Module):
    def __init__(self, cfg):
        super(HuberLoss, self).__init__()
        self.beta = cfg["beta"]
        self.rec_weight = cfg["rec_weight"]
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss(outputs / self.beta, targets / self.beta) * self.beta
        return final_loss, self.rec_weight
 
class MSELoss(nn.Module):
    def __init__(self, cfg):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.rec_weight = cfg["rec_weight"]
    
    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        loss = torch.mean(loss)
        return loss, self.rec_weight

class LambdaRCXYZLoss(nn.Module):
    pass
    # def __init__(self, cfg):
    #     super(LambdaRCXYZLoss, self).__init__()
    #     self.criterion = nn.MSELoss()
    #     self.rec_weight = cfg["rec_weight"]
    
    # def forward(self, outputs, targets):
    #     loss = self.criterion(outputs, targets)
    #     loss = torch.mean(loss)
    #     return loss, self.rec_weight

class LambdaVELLoss(nn.Module):
    pass
    # def __init__(self, cfg):
    #     super(LambdaVELLoss, self).__init__()
    #     self.criterion = nn.MSELoss()
    #     self.rec_weight = cfg["rec_weight"]
    
    # def forward(self, outputs, targets):
    #     loss = self.criterion(outputs, targets)
    #     loss = torch.mean(loss)
    #     return loss, self.rec_weight

class LambdaFCLoss(nn.Module):
    pass
    # def __init__(self, cfg):
    #     super(LambdaFCLoss, self).__init__()
    #     self.criterion = nn.MSELoss()
    #     self.rec_weight = cfg["rec_weight"]
    
    # def forward(self, outputs, targets):
    #     loss = self.criterion(outputs, targets)
    #     loss = torch.mean(loss)
    #     return loss, self.rec_weight
    
class LambdaL2Loss(nn.Module):
    def __init__(self, cfg):
        super(LambdaL2Loss, self).__init__()
        self.criterion = nn.MSELoss(reduction="none")
        self.l2_loss = lambda a, b: (a - b) ** 2
        self.rec_weight = cfg["rec_weight"]
    
    def forward(self, output_, target_):
        assert output_.keys() == target_.keys(), "output and target must have the same keys!"
        output = {k: torch.empty_like(v).copy_(v) for k, v in output_.items()}
        target = {k: torch.empty_like(v).copy_(v) for k, v in target_.items()}
        loss = {}
        for k in output.keys():
            assert output[k].shape == target[k].shape, "output and target must have the same shapes!"
            output[k] = einops.rearrange(output[k], 'b c h -> b h c').unsqueeze(2)
            target[k] = einops.rearrange(target[k], 'b c h -> b h c').unsqueeze(2)
            _loss = _sum_flat(self.l2_loss(output[k], target[k]))
            total_elements = output[k].shape[1] * output[k].shape[2] * output[k].shape[3]
            loss[k] = _loss / total_elements
        return loss, self.rec_weight

def get_loss_fn(loss_cfg):
    loss_name = loss_cfg["name"]
    if loss_name == "huber":
        return HuberLoss(loss_cfg)
    elif loss_name == "mse":
        return MSELoss(loss_cfg)
    elif loss_name == "lrcxyz":
        return LambdaRCXYZLoss(loss_cfg)
    elif loss_name == "lvel":
        return LambdaVELLoss(loss_cfg)
    elif loss_name == "lfc":
        return LambdaFCLoss(loss_cfg)
    elif loss_name == "ll2":
        return LambdaL2Loss(loss_cfg)
    else:
        raise NotImplementedError(f"Loss {loss_name} not implemented.")
    
def _sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))