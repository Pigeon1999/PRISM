import torch 
import torch.nn as nn
import torch.nn.functional as F

# RevIN
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class Stream(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(Stream, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.get_feature = nn.Linear(seq_len * 2, seq_len)

        self.fc1 = nn.Linear(seq_len * 2, seq_len * 2)

        self.pool1_avg = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(seq_len)

        self.pool2_avg = nn.AvgPool1d(kernel_size=4)
        self.ln2 = nn.LayerNorm(seq_len // 2)

        self.pool3_avg = nn.AvgPool1d(kernel_size=6)
        self.ln3 = nn.LayerNorm(seq_len // 3)

        self.fc2 = nn.Linear(seq_len + seq_len // 2 + seq_len // 3, seq_len + seq_len // 2 + seq_len // 3)
        self.get_output = nn.Linear(seq_len + seq_len // 2 + seq_len // 3, pred_len)

    def forward(self, x):
        slope = x[:, :, 1:] - x[:, :, :-1]
        slope = F.pad(slope, (1, 0))  # Δx

        accel = slope[:, :, 1:] - slope[:, :, :-1]
        accel = F.pad(accel, (1, 0))  # Δ²x 
        x_aug = torch.cat([slope, accel], dim=-1)

        feature = self.get_feature(x_aug)
        output = self.fc1(torch.cat([x, feature], dim=-1))
        # output = torch.cat([x, feature], dim=-1)

        out1 = self.pool1_avg(output)
        out1 = self.ln1(out1)

        out2 = self.pool2_avg(output) 
        out2 = self.ln2(out2)

        out3 = self.pool3_avg(output)
        out3 = self.ln3(out3)

        output = self.fc2(torch.cat([out1, out2, out3], dim=-1))
        output = self.get_output(output)

        return output
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Default 336
        self.pred_len = configs.pred_len # Default 96
        self.channels = configs.channels

        # Normalization
        self.revin = RevIN(self.channels, affine=True, subtract_last=False)

        # Stream
        self.stream = Stream(self.seq_len, self.pred_len)

    def forward(self, x, *args):
        # Shape --> (Batch, Input, Channel)
        x = self.revin(x, "norm")
        x = x.permute(0, 2, 1)
        
        output = self.stream(x)
        
        output = output.permute(0, 2, 1)
        output = self.revin(output, "denorm")
        
        return output