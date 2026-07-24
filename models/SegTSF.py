import torch
import torch.nn as nn

class TrendBranch(nn.Module):
    def __init__(self, seq_len, pred_len, channels, period_len, seg_len_x, seg_len_y):
        super(TrendBranch, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = channels
        self.period_len = period_len
        self.seg_len_x = seg_len_x # 입력 세그먼트 길이
        self.seg_len_y = seg_len_y # 출력 세그먼트 길이

        self.seg_num_x = self.seq_len // self.period_len # 입력 하위시퀀스 길이
        self.seg_num_y = self.pred_len // self.period_len # 출력 하위시퀀스 길이
    
        self.seg_num_x2 = self.seg_num_x // self.seg_len_x # 입력 하위시퀀스의 세그먼트 갯수
        self.seg_num_y2 = self.seg_num_y // self.seg_len_y # 출력 하위시퀀스의 세그먼트 갯수 
        
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.period_inner = nn.Linear(self.period_len, self.period_len, bias=False)
        
        if self.seg_len_x == self.seg_num_x and self.seg_num_y == self.seg_len_y:
            self.linear3 = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        else:
            self.linear = nn.Linear(self.seg_len_x, self.seg_len_y, bias=False) # (-1, 세그먼트 갯수, 세그먼트 길이)
            self.linear2 = nn.Linear(self.seg_num_x2, self.seg_num_y2, bias=False) # (-1, 세그먼트 길이, 세그먼트 갯수)
  
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # 관계 학습
        x = x.permute(0, 2, 1)
        x = self.period_inner(x)
        x = x.permute(0, 2, 1)

        # 추세 학습
        if self.seg_len_x == self.seg_num_x and self.seg_num_y == self.seg_len_y:
            x = self.linear3(x)
        else:
            x = x.reshape(-1, self.seg_num_x2, self.seg_len_x)
            x = self.linear(x)
            x = x.permute(0, 2, 1)
            x = self.linear2(x)
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        
        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = x.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        
        return y

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.channels
        self.period_len = configs.period_len # w
        self.seg_len_x = configs.seg_len_x
        self.seg_len_y = configs.seg_len_y
        
        self.trendbranch = TrendBranch(seq_len=self.seq_len,
                                       pred_len=self.pred_len,
                                       channels=self.enc_in,
                                       period_len=self.period_len,
                                       seg_len_x=self.seg_len_x,
                                       seg_len_y=self.seg_len_y)

    def forward(self, x):
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        y = self.trendbranch(x)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y