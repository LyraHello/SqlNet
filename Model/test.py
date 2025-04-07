# test.py
import torch
from .utils import build_dataset
from .sqlNet import SQLNet

def simple_test_model():
    # 加载测试数据
    test_data, _, _ = build_dataset(split="train")  # 仅示例, 你可以换成"test"
    print(f"测试集数据量: {len(test_data)}")

    # 实例化模型 & 加载参数
    vocab_size = 5000  # 你需要和训练时一致
    model = SQLNet(vocab_size=vocab_size, embed_size=128, hidden_size=128)
    model.load_state_dict(torch.load("sqlnet_model.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 简单测试前5条
    for i, data in enumerate(test_data[:5]):
        q_indices = torch.tensor(data["question_indices"], dtype=torch.long, device=device)
        col_indices_list = data["column_token_indices"]
        if len(col_indices_list) == 0:
            continue
        with torch.no_grad():
            select_idx_pred, agg_idx_pred, select_scores = model(q_indices, col_indices_list)
        # 打印预测
        print(f"\nSample {i+1}")
        print("Question tokens:", data["question_tokens"])
        print("Predicted SELECT col idx:", select_idx_pred)
        print("Predicted AGG idx:", agg_idx_pred)
        print("SQL:", data["sql"])

if __name__ == "__main__":
    simple_test_model()
