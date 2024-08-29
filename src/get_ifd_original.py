import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction="none")
device = "cuda"
flag = True

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Input:\n{input}\n\n### Response:"
    )
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default="")
    parser.add_argument("--json_save_path", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--adapter_name_or_path", type=str, default=None)

    args = parser.parse_args()
    return args


def print_info(info, info_name=""):
    print("------------------------------")
    print(f"{info_name}: {info}")
    print("------------------------------")


def check_and_create_directory(path):
    """
    检查指定的路径是否存在，如果不存在则创建该目录。

    参数:
    path (str): 需要检查或创建的目录路径。

    返回:
    bool: 如果路径存在或成功创建，返回True；如果发生错误，返回False。
    """
    if os.path.exists(os.path.dirname(path)):
        print(f"目录 '{os.path.dirname(path)}' 已存在。")
        return True
    else:
        try:
            os.makedirs(os.path.dirname(path))
            print(f"目录 '{os.path.dirname(path)}' 已成功创建。")
            return True
        except OSError as e:
            print(f"创建目录 '{os.path.dirname(path)}' 时发生错误: {e}")
            return False


def print_nvidia():
    import subprocess

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv",
    ]

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    stdout, stderr = process.communicate()

    print("NVIDIA-SMI Output (CSV):")
    print(stdout)


def concat_files(talks, filename):
    talks = sorted(talks, key=lambda x: x["score"])
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(talks, f, ensure_ascii=False)


def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids.contiguous())
    loss = outputs.loss
    perplexity = torch.exp(loss)

    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to("cpu"), sentence_embedding.to("cpu")


def get_perplexity_and_embedding_part_text(
    tokenizer, model, text, target_span, max_length
):

    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)

    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i - 1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    return perplexity.to("cpu"), 0, losses


def get_loss_part_text(tokenizer, text, target_span, max_length, loss_list_):

    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to("cpu")
    start_index = text.rfind(target_span)
    text_temp = text[:start_index]
    token_id_temp = tokenizer.encode(text_temp)
    start_token = len(token_id_temp)
    end_token_real = input_ids.shape[1]

    loss_list = loss_list_[start_token - 1 : end_token_real - 1]

    return (
        end_token_real - start_token,
        input_ids[0][start_token:end_token_real],
        np.array(loss_list),
    )


def get_QA_ifd_score(data_i, tokenizer, model, max_length=1024):
    input_i = data_i["input"]
    output_i = data_i["output"]

    direct_answer_text = "### Response:" + output_i

    temp_dict = {"input": input_i}
    promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)
    whole_text = promt_to_use + output_i
    instruct_i = promt_to_use

    instruct_i_input_ids = tokenizer.encode(
        instruct_i, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    instruct_i_len = instruct_i_input_ids.shape[1]

    ppl_out_alone, _, loss_list_alone = get_perplexity_and_embedding_part_text(
        tokenizer, model, direct_answer_text, output_i, max_length - instruct_i_len + 4
    )
    ppl_out_condition, _, loss_list_condition = get_perplexity_and_embedding_part_text(
        tokenizer, model, whole_text, output_i, max_length
    )

    loss_1_list = loss_list_alone
    loss_2_list = loss_list_condition

    instruct_i_input_ids = tokenizer.encode(
        instruct_i, return_tensors="pt", truncation=True, max_length=max_length
    ).to("cpu")
    instruct_i_len = instruct_i_input_ids.shape[1]

    len_1, token_ids_1, loss_list_1 = get_loss_part_text(
        tokenizer,
        direct_answer_text,
        output_i,
        max_length - instruct_i_len + 4,
        loss_1_list,
    )
    len_2, token_ids_2, loss_list_2 = get_loss_part_text(
        tokenizer, whole_text, output_i, max_length, loss_2_list
    )
    global flag
    if flag:
        print(f"whole_text:{whole_text}\ndirect_answer_text:{direct_answer_text}")
        flag = False

    mean_1 = loss_list_1.mean()
    mean_2 = loss_list_2.mean()
    mean_rate = mean_2 / mean_1
    if pd.isna(mean_rate):
        return 1

    return mean_rate


def main():
    print(f"gpus: {torch.cuda.device_count()}")
    args = parse_args()
    print(f"args: {args}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    with open(args.json_data_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(json_data)
    sampled_data = json_data[start_idx:end_idx]

    print_info(sampled_data[0], "Infer datas: ")

    total = len(sampled_data)
    with tqdm(total=total, desc="IFD-Score") as bar:
        for i in range(total):
            data_i = sampled_data[i]
            sampled_data[i]["score"] = get_QA_ifd_score(data_i, tokenizer, model)
            if i % 50 == 0:
                print_info((args.start_idx), "Start idx")
                print_nvidia()
                bar.update(50)

    check_and_create_directory(args.json_save_path)
    concat_files(sampled_data, args.json_save_path)


if __name__ == "__main__":
    main()
