# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from Model.module.utils import build_dataset, save_vocab
from Model.module.sqlNet import SQLNet

def main():
    # 1. 加载数据并构建词汇表
    train_data, word2idx, idx2word = build_dataset(split="train")
    print(f"训练数据量: {len(train_data)} 条")
    print(f"词表大小: {len(word2idx)}")

    # 2. 实例化模型
    vocab_size = len(word2idx)
    model = SQLNet(vocab_size=vocab_size, embed_size=128, hidden_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. 定义优化器和损失函数（示例: SELECT列使用CrossEntropy）
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_fn_select = nn.CrossEntropyLoss()
    loss_fn_agg = nn.CrossEntropyLoss()

    # 为简化，这里假设: SELECT列只有1个正确列, 聚合函数只有1个正确聚合
    # 训练时需根据 data["sql"] 提取真实标签
    def parse_sql_label(sql_str, col_names):
        """
        简易示例: 解析SQL语句，找出SELECT列和聚合函数(若无则NONE)
        """
        # 可能形如: SELECT COUNT(name) WHERE ...
        # 我们提取 SELECT 后第一个词，判断是不是聚合
        # 再提取列名
        tokens = sql_str.strip().split()
        agg_funcs = ["NONE", "MAX", "MIN", "COUNT", "SUM", "AVG"]
        if len(tokens) < 2:
            # 没有明确SELECT语法，给个默认
            return 0, 0
        # 第一个词是SELECT
        second = tokens[1].upper().strip("(),")
        # 判断是否聚合
        if second in agg_funcs:
            agg_idx = agg_funcs.index(second)
            # 第三个词或同一个词可能是列名
            if len(tokens) >= 3:
                col_candidate = tokens[2].strip("(),")
            else:
                col_candidate = "?"
        else:
            # 没匹配到聚合
            agg_idx = 0
            col_candidate = second.strip("(),")

        # 找col_candidate在 col_names 的索引
        if col_candidate in col_names:
            select_idx = col_names.index(col_candidate)
        else:
            select_idx = 0
        return select_idx, agg_idx

    # 4. 训练循环
    num_epochs = 3
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        for data in train_data:
            question_indices = data["question_indices"]
            col_indices_list = data["column_token_indices"]
            sql_str = data["sql"]
            col_names = data["column_names"]

            if len(col_indices_list) == 0:
                # 若无列，跳过
                continue

            select_idx_true, agg_idx_true = parse_sql_label(sql_str, col_names)

            # 转tensor
            question_tensor = torch.tensor(question_indices, dtype=torch.long, device=device)
            # forward
            select_idx_pred, agg_idx_pred, select_scores = model(question_tensor, col_indices_list)
            # 计算loss
            # select_scores: [num_cols], true是标量 => CrossEntropy需要 [1, num_cols] vs [1]
            select_scores_2d = select_scores.unsqueeze(0)  # [1, num_cols]
            true_select_idx_tensor = torch.tensor([select_idx_true], device=device)
            select_loss = loss_fn_select(select_scores_2d, true_select_idx_tensor)

            # agg预测 -> 需要你自己在 forward 中返回 agg_logits，这里我们暂时只用了pred
            # 为简化，这里就构造一个假的logits, 你可以改sqlNet.py让它返回 agg_logits
            # aggregator的训练类似:
            # agg_loss = loss_fn_agg(agg_logits_2d, true_agg_idx_tensor)

            # 这里先不计算 agg_loss, 仅做 SELECT 列
            loss = select_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch}, Loss = {avg_loss:.4f}")

    # 5. 保存模型与词表
    torch.save(model.state_dict(), "sqlnet_model.pth")
    save_vocab(word2idx, idx2word, "vocab")
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    main()
