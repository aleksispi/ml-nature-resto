import torch
device = 'cpu'
# Uncomment this to use CUDA acceleration if available
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
dtype_torch = torch.float32
from torch import nn, Tensor
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class Network(nn.Module):
    "A generic network with optional L2 regularization"
    def __init__(self, seqstack : OrderedDict, *, l2regularization=None):
        """
        Args:
            seqstack (OrderedDict): The layers of the network.
            l2regularization (float or dict, optional): Regularization strength for all linear layers or for named layers.
        """
        super().__init__()

        self.layer_stack = nn.Sequential(seqstack)
        if isinstance(l2regularization, dict):
            self.l2regularization = [(lambd, seqstack[name]) for name, lambd
                                     in l2regularization.items()]
        elif l2regularization:
            self.l2regularization = [
                (l2regularization, layer) for layer in seqstack.values()
                if isinstance(layer, nn.Linear)]
        else:
            self.l2regularization = None

    def forward(self, x):
        "Apply the network stack on some input"
        return self.layer_stack(x)

    def regularization_loss(self):
        "Compute the total regularization cost"
        if self.l2regularization is None:
            return 0
        loss = 0
        for lambd, layer in self.l2regularization:
            loss = loss + lambd * torch.norm(layer.weight, p=2)
        return loss

    def get_layer(self, name : str):
        "Get layer by name"
        module_dict = dict(self.layer_stack.named_modules())
        return module_dict[name]

    def predict(self, input_data):
        """
        Apply the network on a set of input data.

        Args:
            input_data (np.ndarray or Tensor): Input data

        Returns:
            pred (np.ndarray or Tensor): Predicted output.
        """
        self.eval()
        if isinstance(input_data, np.ndarray):
            inp = torch.tensor(input_data, dtype=dtype_torch, device=device)
            with torch.no_grad():
                pred = self(inp)
            return pred.cpu().detach().numpy()
        with torch.no_grad():
            return self(input_data.to(device))

    def __str__(self):
        s = super().__str__()
        ps = ["Named parameters:"] + [
            f"{name}: {param.numel()}" for name, param in
             self.layer_stack.named_parameters() if param.requires_grad]
        totp = sum(p.numel() for p in self.layer_stack.parameters() if p.requires_grad)
        return s + f"\nTrainable parameters: {totp}\n" + "\n  ".join(ps) + "\n"
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(groups, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(groups, channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = F.leaky_relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + identity
        out = F.leaky_relu(out)

        return out
    
class ResidualDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.gn1   = nn.GroupNorm(groups, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2   = nn.GroupNorm(groups, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = self.skip(x)

        out = F.leaky_relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        out = out + identity
        return F.leaky_relu(out)
    
       
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grl(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)

class AdversarialGenesis(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes, num_years):
        super().__init__()
        self.backbone = backbone   # everything up to flatten

        # task head (your original)
        self.task_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )

        # adversarial head
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_years)
        )

    def forward(self, x, lambda_=0.0):
        features = self.backbone(x)

        task_logits = self.task_head(features)

        # GRL branch
        rev_features = grl(features, lambda_)
        domain_logits = self.domain_head(rev_features)

        return task_logits, domain_logits