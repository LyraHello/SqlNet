import torch
import torch.nn as nn
import torch.nn.functional as F

class AggPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_classes=6):
        super(AggPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, num_classes)  # 双向 LSTM 输出拼接

    def forward(self, x, lengths):
        # x: [batch_size, seq_len, input_size]
        # lengths: [batch_size]
        # 这里可以添加 pack_padded_sequence 处理可变长序列
        out, _ = self.lstm(x)
        # 取最后时刻的输出（或均值），此处简单取均值
        out_mean = torch.mean(out, dim=1)
        logits = self.fc(out_mean)
        return logits
