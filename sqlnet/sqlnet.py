# sqlnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SQLNet(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128):
        super(SQLNet, self).__init__()
        self.hidden_size = hidden_size

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # 问题编码器 (BiLSTM)
        self.question_encoder = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)

        # 列名编码器 (单向LSTM)
        self.column_encoder = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=False)

        # SELECT列预测 (示例：只选1列 -> 多类分类)
        # 这里用 attention 机制：对问题和列做简单attention，然后输出得分
        self.select_score = nn.Linear(hidden_size + 2*hidden_size, 1)

        # 聚合预测 (NONE, MAX, MIN, COUNT, SUM, AVG) -> 6类
        self.agg_classifier = nn.Linear(2*hidden_size, 6)

    def forward(self, q_indices, col_indices_list):
        """
        q_indices: [Q_len], LongTensor (batch_size=1简化)
        col_indices_list: list of list, 每个列的token索引序列
        """
        device = next(self.parameters()).device

        # 问题嵌入与编码
        q_emb = self.embedding(q_indices.unsqueeze(0))  # [1, Q_len, embed_size]
        # LSTM初始化
        bsz = 1
        h0 = torch.zeros(2, bsz, self.hidden_size, device=device)
        c0 = torch.zeros(2, bsz, self.hidden_size, device=device)
        q_enc_out, (h_n, c_n) = self.question_encoder(q_emb, (h0, c0))
        # q_enc_out: [1, Q_len, 2*hidden_size]
        q_enc_out = q_enc_out.squeeze(0)  # [Q_len, 2*hidden_size]
        # 全局问题向量: 取最后时刻正反向拼接
        q_vec = torch.cat([h_n[0], h_n[1]], dim=-1).squeeze(0)  # [2*hidden_size]

        # 列编码
        col_vecs = []
        for col_idxs in col_indices_list:
            col_tensor = torch.tensor(col_idxs, dtype=torch.long, device=device).unsqueeze(0)  # [1, col_len]
            _, (col_h, _) = self.column_encoder(self.embedding(col_tensor))
            # col_h: [1, bsz=1, hidden_size], 取最后时刻
            col_vec = col_h.squeeze(0).squeeze(0)  # [hidden_size]
            col_vecs.append(col_vec)
        if len(col_vecs) == 0:
            # 若无列，直接返回空值，防止 stack 报错
            return None, None, None

        col_vecs = torch.stack(col_vecs)  # [num_cols, hidden_size]

        # Attention: 对每个列，计算与问题的注意力向量
        att_question_vecs = []
        for col_vec in col_vecs:
            # expand col_vec to [Q_len, hidden_size]
            col_expand = col_vec.unsqueeze(0).expand(q_enc_out.size(0), -1)
            concat = torch.cat([col_expand, q_enc_out], dim=1)  # [Q_len, hidden_size+2*hidden_size]
            scores = self.select_score(concat).squeeze(1)  # [Q_len]
            alpha = F.softmax(scores, dim=0)
            att_vec = torch.sum(alpha.unsqueeze(1) * q_enc_out, dim=0)  # [2*hidden_size]
            att_question_vecs.append(att_vec)
        att_question_vecs = torch.stack(att_question_vecs)  # [num_cols, 2*hidden_size]

        # SELECT列预测: 计算对每列的得分
        # 拼接 col_vec( hidden_size ) + att_question_vec(2*hidden_size )
        select_input = torch.cat([col_vecs, att_question_vecs], dim=1)  # [num_cols, 3*hidden_size]
        select_scores = self.select_score(select_input).squeeze(1)      # [num_cols]
        select_prob = F.softmax(select_scores, dim=0)
        select_idx_pred = torch.argmax(select_prob).item()

        # 聚合预测: 用 q_vec -> [2*hidden_size], 映射到6类
        agg_logits = self.agg_classifier(q_vec)
        agg_prob = F.softmax(agg_logits, dim=-1)
        agg_idx_pred = torch.argmax(agg_prob).item()

        return select_idx_pred, agg_idx_pred, select_scores

    # TODO:可以定义一个loss函数
