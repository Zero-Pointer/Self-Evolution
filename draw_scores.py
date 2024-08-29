import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--father_dir", type=str, default='.')
    parser.add_argument("-r", action='store_true')
    parser.add_argument("-w", type=str, default="")
    parser.add_argument("-o", type=str, default="")

    args = parser.parse_args()
    return args

args = parse_args()
if args.o == "":
    t_dir = args.father_dir
    if t_dir[-1] == '/':
        t_dir = t_dir[:-1]
    args.o = t_dir.split('/')[-1]

def list_files(directory):
    predicted = []
    for root, dirs, files in os.walk(directory):
        t = []
        for file in files:
            if file.endswith('all_results.json'):
                t.append(os.path.join(root, file))
        predicted.extend(t)
    return predicted

def get_all_score(filepath):
    all_score = 0
    scores = []
    jump_count = 0
    with open(os.path.join(filepath, 'all_results.json'), encoding='utf-8') as f:
        datas = json.load(f)
    if 'judge' not in datas.keys():
        datas['judge'] = 0
    scores = [datas['predict_bleu-4'], datas['predict_rouge-1'], datas['predict_rouge-2'], datas['predict_rouge-l']]
    return scores

def check_word_in_string(s):
    words = args.w
    if words == "":
        return True
    else:
        words = words.split(',')
    
    # 遍历单词集合，检查字符串s中是否包含这些单词
    for word in words:
        if word in s:
            return True  # 如果找到匹配的单词，立即返回True
    
    # 如果没有找到匹配的单词，返回False
    return False

def sort_dataframe_columns(df):
    """
    对DataFrame的列按照列名进行排序。

    参数:
    df (pd.DataFrame): 输入的DataFrame。

    返回:
    pd.DataFrame: 列按照列名字典序排列的新DataFrame。
    """
    # 获取输入DataFrame的列名，并按照字典序排序
    sorted_columns = sorted(df.columns, key=lambda x: str(x).lower())

    # 使用排序后的列名重新索引DataFrame，以按照指定顺序排列列
    sorted_df = df[sorted_columns]

    return sorted_df

# 定义一个自定义的排序函数
def sort_key(col_name):
    match = re.search(r'_(\d+)$', col_name)
    if match:
        return int(match.group(1))
    return float('inf')  # 如果没有匹配到数字，则将其放在最后

def add_baseline_data(df):

    # 在第一列插入数据
    # Qwen72B
    new_col_first = pd.DataFrame({'Base': [7.26521, 27.36001900000001, 8.097568, 15.389878]}, index=df.index)
    df = pd.concat([new_col_first, df], axis=1)

    # 在最后一列插入数据
    # Qwen72B
    new_col_last = pd.DataFrame({'Qwen72B-Data': [8.899804,  29.015283000000004, 9.651207000000001, 17.519384]}, index=df.index)
    df = pd.concat([df, new_col_last], axis=1)
    return df

if __name__ == '__main__':
    files = list_files(args.father_dir)
    print(files)
    scores = {}
    for file in tqdm(files):
        file = file.split('all_results.json')[0]
        model = file.split('/')[-2].replace('mobile_', '')
        if not check_word_in_string(model):
            continue
        scores[model] = get_all_score(file)
    scores = dict(sorted(scores.items(), key=lambda item: int(re.search(r'[-_](\d+)$', item[0]).group(1))))
    df = pd.DataFrame(scores)
    
    # 使用自定义的排序函数对列名进行排序
    sorted_columns = sorted(df.columns, key=sort_key)
    # 使用排序后的列名重新索引DataFrame
    df = df[sorted_columns]
    
    df.index = ['BLEU4', 'ROUGE1', 'ROUGE2', 'ROUGEL']
    
    df = add_baseline_data(df)
    
    df.to_csv(f'{args.o}.csv')
    # 创建一个Figure对象，并设置图像尺寸为宽10英寸、高6英寸
    fig = plt.figure(figsize = (10, 6))

    # 添加一个子图（Axes对象）
    ax = fig.subplots()
    # 在子图上绘制图表
    ax = df.plot(kind='bar', ax=ax, title=args.father_dir, cmap='tab20')
    num1 = 1.0
    num2 = 1.0
    num3 = 0
    num4 = 0.1
    ax.legend(bbox_to_anchor = (num1, num2), loc = num3, borderaxespad = num4)
    ax.set_xlabel('Scores', rotation=0)
    ax.tick_params(rotation=0)
    plt.tight_layout() 
    plt.savefig(f"{args.o}.png", format="png", dpi=400)
