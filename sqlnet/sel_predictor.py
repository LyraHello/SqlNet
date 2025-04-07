import torch
import torch.nn as nn
import torch.nn.functional as F

class SelPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(SelPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 输出每个候选列的得分
        self.fc = nn.Linear(hidden_size*3, 1)  # 拼接列向量和问题向量

    def forward(self, q_enc, q_lengths, col_enc, col_vectors):
        """
        q_enc: 问题的编码结果，[batch_size, Q_len, 2*hidden_size]
        col_enc: 列的编码结果，[batch_size, max_col_num, hidden_size]
        col_vectors: 单个列的表示（例如经过单向 LSTM 编码），[batch_size, max_col_num, hidden_size]
        """
        # 这里简化处理：假设我们对每个列与问题计算注意力
        # 最后输出每个列的得分
        batch_size, num_cols, _ = col_vectors.size()
        # 对每个列，简单拼接其表示和问题全局向量（如均值），计算得分
        q_global = torch.mean(q_enc, dim=1)  # [batch_size, 2*hidden_size]
        q_global_exp = q_global.unsqueeze(1).expand(-1, num_cols, -1)  # [batch_size, num_cols, 2*hidden_size]
        combined = torch.cat([col_vectors, q_global_exp], dim=2)  # [batch_size, num_cols, 3*hidden_size]
        scores = self.fc(combined).squeeze(2)  # [batch_size, num_cols]
        return scores
