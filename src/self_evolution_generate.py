import json
import re
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import torch
import json
import argparse
from tqdm import tqdm
from peft import PeftModel
import fcntl
import time
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="0")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--target_file", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--use_vllm", action="store_true")

    args = parser.parse_args()
    return args


args = parse_args()
print(f"args: {args}")


def print_info(info, info_name=""):
    print("------------------------------")
    print(f"{info_name}: {info}")
    print("------------------------------")


def clean_string(input_string, blacklist=["问题："]):
    for item in blacklist:
        input_string = input_string.replace(item, "")

    pattern = re.compile(r"^.*问题可以是：", re.DOTALL)
    input_string = re.sub(pattern, "", input_string)

    pattern = re.compile(r"^.*问题是：", re.DOTALL)
    input_string = re.sub(pattern, "", input_string)

    input_string = input_string.strip("\n")

    input_string = input_string.strip('"')

    input_string = input_string.replace("?", "？")
    input_string = input_string.split("？")[0] + "？"

    return input_string


def append_ins_answer(example):
    ques_prompt = """你是一名通信领域的专家，负责回答各种通信领域问题。你必须根据需求生成响应。
- Workflow:
  1. 接收并解析用户的问题。
  2. 阅读并分析用户提供的文档。
  3. 结合自身知识和文档内容给出简洁且全面的回答。
- Examples:
  问题：太阳系中最大的行星是哪一颗？
  知识片段：太阳系由八大行星组成，其中木星是最大的行星，它的质量是其他所有行星总和的2.5倍。
  回答：太阳系中最大的行星是木星，它的质量是其他所有行星总和的2.5倍。
  
  问题：什么是光合作用？
  知识片段：光合作用是植物、藻类和某些细菌利用太阳光能将水和二氧化碳转化为葡萄糖和氧气的过程。
  回答：光合作用是植物、藻类和某些细菌通过太阳光能将水和二氧化碳转化为葡萄糖和氧气的过程。
- Warning:
  1. 你的答案在生成之后会与文档独立发送，所以不要在回答中出现“文档”的字样，不然会困惑用户，因为用户看不见对应的文档，只能读到你的回答。
  2. 你的回答一定要保证两点：简洁和准确。
文档: 
"""
    if type(example["instruction"]) == str:
        example["instruction"] = ques_prompt + example["instruction"]
        example["input"] = "问题：" + example["input"]
    else:
        for i in range(len(example["instruction"])):
            example["instruction"][i] = ques_prompt + example["instruction"][i]
            example["input"][i] = "问题：" + example["input"][i]
    return example


def append_ins_question(example):
    ques_prompt = """参考文档: {}
你是一个通信领域的专家，请综合你所有的知识和以上信息，提出一个可以被上述文档中知识回答的问题。
注意事项1: 问题要尽量简洁。
注意事项2: 除了问题，不要输出其他任何内容。
注意事项3: 不允许大片的复述知识片段的内容。
注意事项4: 问题中不可以出现“参考文档”字样。
注意事项5: 一个问题中不要包含多个子问题，只能包含一个问题。
注意事项6: 不要输出陈述句，必须是一个问题！
接下来请你提出一个问题
\n问题："""
    if type(example["instruction"]) == str:
        example["instruction"] = ques_prompt.format(example["instruction"])
    else:
        for i in range(len(example["instruction"])):
            example["instruction"][i] = ques_prompt.format(example["instruction"][i])
    return example


def merge_input_ins_answer(example):
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}\n{}<|im_end|>\n<|im_start|>assistant\n"
    if type(example["instruction"]) == str:
        example["question"] = prompt.format(example["instruction"], example["input"])
    else:
        example["question"] = []
        for i in range(len(example["instruction"])):
            example["question"].append(
                prompt.format(example["instruction"][i], example["input"][i])
            )
    return example


def merge_input_ins_question(example):
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    if type(example["instruction"]) == str:
        example["question"] = prompt.format(example["instruction"])
    else:
        example["question"] = []
        for i in range(len(example["instruction"])):
            example["question"].append(prompt.format(example["instruction"][i]))
    return example


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


def convert_dataset(dataset, type):

    dataset = dataset.map(
        append_ins_answer if type == "answer" else append_ins_question, batched=True
    )
    dataset = dataset.map(
        merge_input_ins_answer if type == "answer" else merge_input_ins_question,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return dataset


def concat_files(obj, f):
    for data in obj:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_to_lock_file(obj, file_path, exclusive=True):

    with open(file_path, "a+", encoding="utf-8") as f:

        while True:
            try:
                if exclusive:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError as e:
                time.sleep(1)
        concat_files(obj, f)

        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def use_transformers_pipeline(type):
    model_name_or_path = args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    print_nvidia()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, padding_side="left"
    )
    model.requires_grad_(False)
    model.eval()
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, model_id=args.adapter)
        model = model.merge_and_unload()
        print(f"merged: {args.adapter}")

    pipe = pipeline(
        model=model,
        task="text-generation",
        tokenizer=tokenizer,
        max_new_tokens=1024,
        truncation=True,
        padding=True,
    )
    print_info("create pipeline", "Stage: ")
    print_nvidia()
    i = 0
    bs = 6
    for out in tqdm(
        pipe(
            KeyDataset(infer_datasets, "question"),
            batch_size=bs,
            return_full_text=False,
        ),
        total=len(infer_datasets),
    ):
        if type == "answer":
            data[i]["output"] = out[0]["generated_text"]
        else:
            data[i]["input"] = out[0]["generated_text"]
        i += 1
        if i % 50 == 0:
            print_info((args.start_idx, bs), "Start idx, BS: ")
            print_nvidia()


def use_vllm_pipeline(type):
    llm = LLM(
        model=args.model_name_or_path,
        dtype="half",
        max_model_len=2048,
        enforce_eager=True if type == "answer" else False,
        gpu_memory_utilization=0.55,
        tensor_parallel_size=args.gpus,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=0.95, max_tokens=2048
    )

    print_info(sampling_params, "SamplingParams")

    outputs = llm.generate(
        KeyDataset(infer_datasets, "question"),
        sampling_params,
    )
    for i in range(len(outputs)):
        output = outputs[i]
        generated_text = output.outputs[0].text
        if type == "answer":
            data[i]["output"] = generated_text
        else:
            data[i]["input"] = clean_string(generated_text)


if __name__ == "__main__":
    print(f"gpus: {torch.cuda.device_count()}")

    infer_file = args.data_file

    evolution_step = args.id

    train_file = args.target_file

    with open(infer_file, encoding="utf-8") as f:
        data = json.load(f)
    if args.end_idx == -1:
        args.end_idx = len(data)
    else:
        args.end_idx = min(len(data), args.end_idx)

    data = data[args.start_idx : args.end_idx]

    infer_datasets = convert_dataset(Dataset.from_list(data), args.type)
    print_info(infer_datasets[0], "Infer datas: ")

    if not args.use_vllm:
        use_transformers_pipeline(args.type)
    else:
        use_vllm_pipeline(args.type)

    write_to_lock_file(data, train_file)
