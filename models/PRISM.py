import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # look-back window length
        self.pred_len = configs.pred_len  # forecast horizon length
        self.channels = configs.channels  # number of input features
        self.period_len = configs.period_len  # length of one period in time series
        self.kernel_size = configs.kernel_size  # convolution kernel size for trend branch
        self.stride = configs.stride  # convolution stride for trend branch
        self.dilation = configs.dilation  # convolution dilation for trend branch
        self.temporal_kernel_size = configs.temporal_kernel_size  # convolution kernel size for temporal mixing
        self.temporal_stride = 1  # convolution stride for temporal mixing ## Fixed
        self.temporal_dilation = 1  # convolution dilation for temporal mixing ## Fixed

        self.seg_num_x = self.seq_len // self.period_len  # number of segments in input sequence
        self.seg_num_y = self.pred_len // self.period_len  # number of segments in output sequence

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation, bias=False) # summarization of each segment
        self.conv_seg_num = (self.seg_num_x - (self.kernel_size - 1) * self.dilation - 1) // self.stride + 1 # number of features after convolution

        self.temporal_mixer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.temporal_kernel_size, stride=self.temporal_stride, dilation=self.temporal_dilation, padding=(self.temporal_dilation * (self.temporal_kernel_size - 1) + 1 - self.temporal_stride) // 2, padding_mode="circular", bias=False)

        self.linear = nn.Linear(self.conv_seg_num, self.seg_num_y, bias=False) # forecasting of each segment

    def forward(self, x):
        B, L, C = x.shape

        # 1. Normalization and permute
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = (x - seq_mean).permute(0, 2, 1) # (B,L,C) -> (B,C,L)

        # 2. Reorganization (BC, P, N)
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1) # (B,C,L) -> (BC, P, N)

        # 3. Sequence summarization
        x = x.reshape(-1, 1, self.seg_num_x) # (BC, P, N) -> (BC * P, 1, N)
        x = self.conv1d(x) # (BC * P, 1, N) -> (BC * P, 1, conv_seg_num)
        x = x.reshape(-1, self.period_len, self.conv_seg_num) # (BC * P, 1, conv_seg_num) -> (BC, P, conv_seg_num)

        # 4 Temporal mixing
        x = x.permute(0, 2, 1).reshape(-1, 1, self.period_len) # (BC, P, conv_seg_num) -> (BC * conv_seg_num, 1, P)
        x = self.temporal_mixer(x) # (BC * conv_seg_num, 1, P) -> (BC * conv_seg_num, 1, P)
        x = x.reshape(-1, self.conv_seg_num, self.period_len).permute(0, 2, 1) # (BC * conv_seg_num, 1, P) -> (BC, P, conv_seg_num)
        
        # 5. Sequence forecasting
        y = self.linear(x) # (BC, P, conv_seg_num) -> (BC, P, seg_num_y)

        # 6. Reconstruction
        y = y.permute(0, 2, 1).contiguous() # (BC, P, seg_num_y) -> (BC, seg_num_y, P)
        y = y.reshape(B, self.channels, self.pred_len) # (BC, seg_num_y, P) -> (B,C,H)
        
        # 7. Denormalization
        y = y.permute(0, 2, 1) + seq_mean # (B,C,H) -> (B,H,C)

        return y