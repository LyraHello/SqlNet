# utils.py
import re
import json
from collections import defaultdict
from datasets import load_dataset

def load_sql_generator_data(split="train"):
    """
    加载AI4DS/sql_generator_no_cot数据集指定切分（train/test等）
    返回datasets.Dataset对象。
    """
    dataset = load_dataset("AI4DS/sql_generator_no_cot", split=split)
    return dataset

def parse_prompt(prompt_text):
    """
    从 prompt 中解析问题、提示和列信息的示例函数
    返回 (full_question, col_names)
    """
    # 提取数据库模式
    schema_match = re.search(r"### Database Schema:(.*)### Question:", prompt_text, re.S)
    schema_text = schema_match.group(1) if schema_match else ""

    # 提取问题
    q_match = re.search(r"### Question:\s*(.*?)(?:Hint:|Please respond)", prompt_text, re.S)
    question = q_match.group(1).strip() if q_match else ""

    # 提取提示
    hint_match = re.search(r"Hint:\s*(.*?)(?:Please respond)", prompt_text, re.S)
    hint = hint_match.group(1).strip() if hint_match else ""

    # 合并问题和提示
    full_question = question
    if hint:
        full_question += " " + hint

    # 从schema_text中解析列名（简单示例）
    col_names = []
    schema_text_no_comment = re.sub(r'--.*', '', schema_text)
    # 匹配形如 "列名 类型"
    for col, col_type in re.findall(r'(\w+)\s+[A-Z]+', schema_text_no_comment):
        if col.lower() == "references":
            continue
        col_names.append(col)

    return full_question.strip(), col_names

def tokenize(text):
    """
    基础分词：按非字母数字字符拆分
    """
    tokens = re.split(r'\W+', text)
    tokens = [t for t in tokens if t]
    return tokens

def build_dataset(split="train"):
    """
    1. 加载指定split的数据
    2. 解析prompt，提取question & columns
    3. 建立词汇表 & 将文本转为索引
    4. 返回一个列表，每个元素包含必要字段(示例)
    """
    dataset = load_sql_generator_data(split)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    word_count = defaultdict(int)

    processed_data = []
    for item in dataset:
        prompt = item["prompt"]
        sql = item["response"]
        q_text, col_names = parse_prompt(prompt)
        q_tokens = tokenize(q_text.lower())
        col_tokens_list = [tokenize(col.lower().replace("_", " ")) for col in col_names]

        # 更新词表
        for tok in q_tokens:
            word_count[tok] += 1
            if tok not in word2idx:
                idx = len(word2idx)
                word2idx[tok] = idx
                idx2word[idx] = tok

        for col_toks in col_tokens_list:
            for tok in col_toks:
                word_count[tok] += 1
                if tok not in word2idx:
                    idx = len(word2idx)
                    word2idx[tok] = idx
                    idx2word[idx] = tok

        # 转索引
        q_indices = [word2idx.get(tok, 1) for tok in q_tokens]
        col_indices_list = [[word2idx.get(t, 1) for t in col_toks] for col_toks in col_tokens_list]

        processed_data.append({
            "question_tokens": q_tokens,
            "question_indices": q_indices,
            "column_names": col_names,
            "column_token_indices": col_indices_list,
            "sql": sql
        })

    return processed_data, word2idx, idx2word

def save_vocab(word2idx, idx2word, path_prefix="vocab"):
    """
    保存词汇表
    """
    with open(f"{path_prefix}_word2idx.json", "w") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)
    with open(f"{path_prefix}_idx2word.json", "w") as f:
        json.dump(idx2word, f, ensure_ascii=False, indent=2)
