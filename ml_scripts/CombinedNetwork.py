import torch
device = 'cpu'
dtype_torch = torch.float32
from torch import nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
from ml_scripts import Network

def compute_cnn_out_dim(cnn, input_image, device="cpu"):
    """
    Computes the flattened output shape of an image passed through the CNN. 
    """
    cnn.eval()
    with torch.no_grad():
        # input_image=input_image.to(device)    # Is this needed maybe?
        out = cnn(input_image)
    return out.view(1, -1).size(1)        

class CNNBlock(nn.Module):
    def __init__(self, layers : OrderedDict):
        super().__init__()
        self.layer_stack = nn.Sequential(layers)

    def forward(self, x):
        "Apply the network stack on some input"
        return self.layer_stack(x)

class CNNLSTM(nn.Module):
    def __init__(self, cnn_layers, hidden_dim, num_layers, num_classes, bidir=False,temp_mode="final"):
        """
        Args:
            cnn_layers:  
            hidden_dim:
            num_layers: 
            H (int): image height
            W (int): image width

        """
        # super.__init__()
        super(CNNLSTM, self).__init__()
        self.num_classes = num_classes                   #<------------------------------------- ?????????????
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidir = bidir
        self.temp_mode=temp_mode
        self.cnn = CNNBlock(cnn_layers)
        #cnn_out_dim = compute_cnn_out_dim(self.cnn,single_im_sample)   #<----------------------------------------------Removed these
        #lstm_in_dim = cnn_out_dim   # TODO: scale factor? e.g. lstm_in_dim = cnn_out_dim*0.5*0.5*H*W
        self.dropout = nn.Dropout(p=0.1)                #<------------------------------------------------------------- ADDED (dropout between CNN and LSTM)
        #self.pool = nn.AdaptiveAvgPool2d((2,2))     #<----------------------- ADDED (instead of GAP)
        
        last_conv = [m for m in self.cnn.modules() if isinstance(m, nn.Conv2d)][-1]   #<---------------------------------added this
        lstm_in_dim = last_conv.out_channels        #*4      #(scale fac 4 is for adaptive avg pooling)

        self.lstm = nn.LSTM(lstm_in_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidir)
        if self.bidir:
                self.fc_in_dim = hidden_dim * 2
        else:
            self.fc_in_dim = hidden_dim
        self.fc = nn.Linear(self.fc_in_dim, num_classes)
        
        self.attn = nn.Sequential(
            nn.Linear(self.fc_in_dim, self.fc_in_dim),
            nn.Tanh(),
            nn.Linear(self.fc_in_dim, 1)
        )

        


    def forward(self, x, seq_lens):
        """
        Apply the network to a batch of data.
        x dim: (B, T, C, H, W)
        seq_lens (Tensor or list): True sequence lengths, shape (B,)
        """
        seq_lens = seq_lens.cpu()   # Move to CPU
        batch_size, T, C, H, W = x.shape
        # Spatial processing with CNN
        c_in = x.reshape(batch_size * T, C, H, W) 
        c_out = self.cnn(c_in)



        #GLOBAL AVERAGE POOLING             <----------ADDED
        c_out = c_out.mean(dim=(-2, -1))
        
        # c_out = self.pool(c_out)      # Testing this instead :)
        # c_out = c_out.flatten(1)


        #c_out = self.dropout(c_out)             # <-------------------- in-between dropout layer

        # LSTM part (temporal)
        r_in = c_out.reshape(batch_size, T, -1)     # Restore T dimension
      
        # Pack the padded input for LSTM processing
        r_in = nn.utils.rnn.pack_padded_sequence(r_in, seq_lens, batch_first=True, enforce_sorted=False)
       
        output, (h_n, _) = self.lstm(r_in) # h_n shape: (num_layers, B, hidden_size)

        # Use the last LSTM layer’s final hidden state
        if self.temp_mode=="final":
            if self.bidir:
                # Forward and backward hidden states
                h_forward = h_n[-2]    # (B, hidden_size)
                h_backward = h_n[-1]   # (B, hidden_size)

                h_final = torch.cat([h_forward, h_backward], dim=1)
            else:    
                h_final = h_n[-1]                 # (B, hidden_size)
            
            logits = self.fc(h_final)          # (B, num_classes)

        else:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

            
            mask = torch.arange(output.size(1), device=output.device)[None, :] < seq_lens.to(output.device)[:, None]
            mask = mask.unsqueeze(-1)

            if self.temp_mode == "mean":
                pooled = (output * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6).float()
                logits = self.fc(pooled)

            elif self.temp_mode == "attention":
                # --- attention scores ---
                scores = self.attn(output)   # (B, T, 1)

                # mask padded positions
                scores = scores.masked_fill(~mask, float('-inf'))

                # softmax over time
                weights = torch.softmax(scores, dim=1)   # (B, T, 1)

                # weighted sum
                weighted = (weights * output).sum(dim=1)   # (B, H)

                logits = self.fc(weighted)


        

       
        return logits
        

    def predict(self, input_data):
        """
        Apply the network on a set of input data.
        """
        self.eval()
        inp = torch.tensor(input_data, dtype=dtype_torch, device=device)
        with torch.no_grad():
            pred = self(inp)
        return pred.cpu().numpy()

    def __str__(self):
        s = super().__str__()
        totp = sum(p.numel() for r in self.rnns for p in r.parameters() if p.requires_grad)
        totp = totp + sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return s + f"\nTrainable parameters: {totp}\n"


