import argparse
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import shutil
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--source_data_file", type=str, required=True)
    parser.add_argument("--identify", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--eval_to_path", type=str, required=True)
    parser.add_argument("--ifd_flag", type=bool, default=True)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--ques_first", type=bool, default=True)
    parser.add_argument("--rewrite", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="even")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--use_sample", type=int, default=100)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=8)
    parser.add_argument("--ifd_k", type=int, default=2000)
    parser.add_argument("--train_epoch", type=int, default=1)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--infer_gpu", type=int, default=1)
    parser.add_argument("--worker_number", type=int, default=1)
    args = parser.parse_args()
    return args


def run_my_process(cmds):
    print(f"cmds: {cmds}")
    try:
        process = subprocess.Popen(
            cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        print(e)
    try:
        while True:
            line = process.stderr.readline()
            if line == "" and process.poll() is not None:
                break
            print(line, end="")
    finally:
        process.stdout.close()


def run_generation(
    generation_id,
    start_id,
    end_id,
    identify,
    type,
    file_path,
    target_file,
    model_path,
):
    cmds = [
        "python",
        f"src/self_evolution_generate.py",
        f"--data_file={file_path}",
        f"--target_file={target_file}",
        f"--type={type}",
        f"--id={generation_id}",
        f"--start_idx={start_id}",
        f"--end_idx={end_id}",
        f"--model_name_or_path={model_path}",
    ]
    if generation_id != 0:
        cmds.append("--adapter")
        cmds.append(f"model/{identify}_generations/generation_{generation_id}")

    print(f"generating {type}...")
    run_my_process(cmds)


def run_generation_vllm(
    generation_id,
    start_id,
    end_id,
    type,
    file_path,
    target_file,
    target_model_path,
    infer_gpu,
):
    cmds = [
        "python",
        f"src/self_evolution_generate.py",
        "--model_name_or_path",
        target_model_path,
        f"--data_file={file_path}",
        f"--target_file={target_file}",
        f"--type={type}",
        f"--id={generation_id}",
        f"--start_idx={start_id}",
        f"--end_idx={end_id}",
        f"--gpus={infer_gpu}",
        "--use_vllm",
        "--temperature",
        "0.5",
    ]

    print(f"generating {type} with vllm...")
    run_my_process(cmds)


def check_file_exists(file_path):
    return os.path.isfile(file_path)


def check_and_create_directory(path):
    if os.path.exists(os.path.dirname(path)):
        return True
    else:
        try:
            os.makedirs(os.path.dirname(path))
            return True
        except OSError as e:
            return False


def merge_sft_model(generation_id, identify, model_path):

    if generation_id == 0:
        return model_path
    export_dir = f"model/{identify}_generations/generation_{generation_id}/full_model"
    if os.path.exists(export_dir):
        print("Full model's path exist!!! No need to merge!!!")
        return export_dir
    cmds = [
        "python",
        "src/export_model.py",
        "--model_name_or_path",
        model_path,
        "--template",
        "qwen",
        "--finetuning_type",
        "lora",
        "--export_dir",
        export_dir,
        "--export_size",
        "4",
        "--export_legacy_format",
        "False",
    ]
    if generation_id != 0:
        cmds.append("--adapter_name_or_path")
        cmds.append(f"model/{identify}_generations/generation_{generation_id}")
    run_my_process(cmds)

    return f"model/{identify}_generations/generation_{generation_id}/full_model"


def delete_merged_model(model_path):
    import os

    folder_path = model_path
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isdir(file_path):
                print(
                    f"There is a folder in folder {folder_path}, will delete nothing!"
                )
                return
        shutil.rmtree(folder_path)
        print(f"The folder {folder_path} has been deleted successfully")
    else:
        print(f"The folder {folder_path} does not exist")


def generage_questions_multithread(
    model_path,
    identify,
    source_data_file,
    generation_id,
    use_vllm=False,
    batch_size=100,
    worker_number=1,
    infer_gpu=1,
):
    file_path = source_data_file
    target_file = f"result_datas/{identify}/qs_train_{identify}.jsonl"

    print("“Stage Generate Question”")

    check_and_create_directory(target_file)
    with open(target_file, "w", encoding="utf-8") as f:
        print("Init target file!")
    with open(file_path, encoding="utf-8") as f:
        length = len(json.load(f))
    with ThreadPoolExecutor(max_workers=worker_number) as executor:
        if use_vllm:
            target_model_path = merge_sft_model(
                generation_id, identify, model_path
            )
            futures = [
                executor.submit(
                    run_generation_vllm,
                    generation_id,
                    i,
                    i + batch_size,
                    "question",
                    file_path,
                    target_file,
                    target_model_path,
                    infer_gpu,
                )
                for i in range(0, length, batch_size)
            ]
        else:
            futures = [
                executor.submit(
                    run_generation,
                    generation_id,
                    i,
                    i + batch_size,
                    identify,
                    "question",
                    file_path,
                    target_file,
                    model_path,
                )
                for i in range(0, length, batch_size)
            ]
        with tqdm(total=max(1, length // batch_size), desc="Processing") as pbar:
            for future in futures:
                pbar.update(1)
    qas = []
    for line in open(target_file, "r", encoding="utf-8"):
        qas.append(json.loads(line))

    os.remove(target_file)

    with open(target_file.replace("jsonl", "json"), "w", encoding="utf-8") as f:
        json.dump(qas, f, ensure_ascii=False)

    return target_file.replace("jsonl", "json")


def generage_answers_multithread(
    model_path,
    identify,
    source_data_file,
    generation_id,
    use_vllm=True,
    ques_first=True,
    rewrite=True,
    batch_size=100,
    worker_number=1,
    infer_gpu=1,
):
    if ques_first:
        file_path = generage_questions_multithread(
            model_path,
            identify,
            source_data_file,
            generation_id,
            use_vllm,
            batch_size,
            worker_number,
            infer_gpu,
        )
    else:
        file_path = source_data_file
    target_file = f"result_datas/{identify}/qas_train_{identify}_{generation_id}.jsonl"

    print("“Stage Generate Answer”")

    if check_file_exists(target_file.replace("jsonl", "json")):
        if not rewrite:
            print(
                "Target file existed! If you want to overwrite it please change code or delete it first! {}".format(
                    target_file.replace("jsonl", "json")
                )
            )
            return

    check_and_create_directory(target_file)
    with open(target_file, "w", encoding="utf-8") as f:
        print("Init target file!")
    with open(file_path, encoding="utf-8") as f:
        length = len(json.load(f))
    with ThreadPoolExecutor(max_workers=worker_number) as executor:
        if use_vllm:
            target_model_path = merge_sft_model(
                generation_id, identify, model_path
            )
            futures = [
                executor.submit(
                    run_generation_vllm,
                    generation_id,
                    i,
                    i + batch_size,
                    "answer",
                    file_path,
                    target_file,
                    target_model_path,
                    infer_gpu,
                )
                for i in range(0, length, batch_size)
            ]
        else:
            futures = [
                executor.submit(
                    run_generation,
                    generation_id,
                    i,
                    i + batch_size,
                    identify,
                    "answer",
                    file_path,
                    target_file,
                    model_path,
                )
                for i in range(0, length, batch_size)
            ]
        with tqdm(total=max(1, length // batch_size), desc="Processing") as pbar:
            for future in futures:
                pbar.update(1)
    qas = []
    for line in open(target_file, "r", encoding="utf-8"):
        qas.append(json.loads(line))

    if use_vllm and generation_id != 0:
        delete_merged_model(target_model_path)

    os.remove(target_file)

    with open(target_file.replace("jsonl", "json"), "w", encoding="utf-8") as f:
        json.dump(qas, f, ensure_ascii=False)


def train_model(
    evolution_step, identify, model_path, train_epoch=1, cuda_device=0
):
    dataset_file = f"result_datas/{identify}/qas_train_{identify}_{evolution_step}.json"

    print("“Stage Train Model”")

    if evolution_step > 0:
        dataset_file += f",result_datas/{identify}/qas_train_{identify}_ifd.json"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_device}"
    cmds = [
        "python",
        "src/train_bash.py",
        "--stage",
        "sft",
        "--do_train",
        "--dataset_file",
        dataset_file,
        "--model_name_or_path",
        model_path,
        "--template",
        "qwen",
        "--finetuning_type",
        "lora",
        "--lora_target",
        "all",
        "--output_dir",
        f"model/{identify}_generations/generation_{evolution_step+1}",
        "--overwrite_cache",
        "--per_device_train_batch_size",
        "4",
        "--gradient_accumulation_steps",
        "4",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "10",
        "--save_steps",
        "1000",
        "--learning_rate",
        "1e-6",
        "--num_train_epochs",
        "{}.0".format(train_epoch),
        "--plot_loss",
        "--fp16",
        "--overwrite_output_dir",
        "--not_use_prompt",
    ]
    if evolution_step != 0:
        cmds.append("--adapter_name_or_path")
        cmds.append(f"model/{identify}_generations/generation_{evolution_step}")

    run_my_process(cmds)


def eval_model(
    model_path,
    evolution_step,
    identify,
    eval_data,
    eval_to_path,
    use_sample=100,
):

    cmds = [
        "python",
        "src/train_bash.py",
        "--stage",
        "sft",
        "--do_predict",
        "--model_name_or_path",
        model_path,
        "--dataset_file",
        eval_data,
        "--template",
        "qwen",
        "--output_dir",
        f"{eval_to_path}/{identify}_generations",
        "--overwrite_cache",
        "--per_device_eval_batch_size",
        "1",
        "--predict_with_generate",
        "--fp16",
        "--not_use_prompt",
    ]
    if evolution_step != 0:
        cmds.append("--finetuning_type"),
        cmds.append("lora"),
        cmds.append("--adapter_name_or_path")
        cmds.append(f"model/{identify}_generations/generation_{evolution_step}")
    if use_sample > 0:
        cmds.append("--max_samples")
        cmds.append(str(use_sample))

    run_my_process(cmds)


def run_all_evals(
    model_path,
    start_id,
    end_id,
    identify,
    eval_data,
    eval_to_path,
    use_sample=100,
    worker_number=1,
):

    print("“Stage Evaluate Model”")
    with ThreadPoolExecutor(max_workers=worker_number) as executor:
        futures = [
            executor.submit(
                eval_model,
                model_path,
                i,
                identify,
                eval_data,
                eval_to_path,
                use_sample,
            )
            for i in range(start_id, end_id)
        ]

        with tqdm(total=end_id - start_id, desc="Processing") as pbar:
            for future in futures:
                pbar.update(1)


def ifd_model(model_path, evolution_step, identify):
    print("“Stage Calculate IFD Score”")
    cmds = [
        "python",
        "src/get_ifd_original.py",
        "--json_data_path",
        f"result_datas/{identify}/qas_train_{identify}_{evolution_step}.json",
        "--json_save_path",
        f"result_datas/{identify}/qas_train_{identify}_{evolution_step}.json",
        "--model_name_or_path",
        model_path,
        "--max_length=2048",
    ]

    process = subprocess.Popen(
        cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    try:
        while True:
            line = process.stderr.readline()
            if line == "" and process.poll() is not None:
                break
            print(line, end="")
    finally:
        process.stdout.close()


def get_added_source_data(evolution_step, identify, mode="even", ifd_k=2000):
    import random

    all_datas = []
    ifd_qa_path = f"result_datas/{identify}/qas_train_{identify}_ifd.json"
    if evolution_step == 0:
        with open(ifd_qa_path, "w", encoding="utf-8") as f:
            json.dump(all_datas, f, ensure_ascii=False)
        return
    for i in range(0, evolution_step):
        last_ifd_file = f"result_datas/{identify}/qas_train_{identify}_{i}.json"
        with open(last_ifd_file, encoding="utf-8") as f:
            datas = json.load(f)
        for i in range(len(datas)):
            if pd.isna(datas[i]["score"]):
                datas[i]["score"] = 1
        all_datas.extend(datas)
    all_datas = sorted(all_datas, key=lambda x: x["score"], reverse=True)
    print(f"length of history data: {len(all_datas)}")
    ifd_gap = len(all_datas) // ifd_k

    t = []
    if mode == "even":
        for i in range(0, len(all_datas), max(1, ifd_gap)):
            t.append(all_datas[i])
    elif mode == "easy":
        t = all_datas[-ifd_k:]
    elif mode == "random":
        if ifd_k > len(all_datas):
            t = all_datas
        else:
            t = random.sample(all_datas, ifd_k)
    elif mode == "all":
        t = all_datas
    else:
        t = all_datas[:ifd_k]

    with open(ifd_qa_path, "w", encoding="utf-8") as f:
        json.dump(t, f, ensure_ascii=False)
    return


def main():
    args = parse_args()
    print(f"args: {args}")

    print(f"Start to run: {args.identify}")

    for i in range(args.start_id, args.end_id):
        print("--------------------------")
        print(f"Generation: {i}")
        print("--------------------------")

        generage_answers_multithread(
            model_path=args.model_path,
            identify=args.identify,
            source_data_file=args.source_data_file,
            generation_id=i,
            use_vllm=args.use_vllm,
            ques_first=args.ques_first,
            rewrite=args.rewrite,
            batch_size=args.batch_size,
            worker_number=args.worker_number,
            infer_gpu=args.infer_gpu,
        )
        # IFD筛选数据模式
        # 1. random - 随机
        # 2. easy - 简单题
        # 3. even - 均匀抽取
        # 4. hard - 困难题
        # 5. all - 所有数据都选，不筛选
        get_added_source_data(
            evolution_step=i, identify=args.identify, mode=args.mode, ifd_k=args.ifd_k
        )

        with ThreadPoolExecutor(max_workers=args.worker_number) as executor:
            futures = []
            futures.append(
                executor.submit(
                    train_model,
                    i,
                    args.identify,
                    args.model_path,
                    args.train_epoch,
                    args.cuda_device,
                )
            )
            futures.append(
                executor.submit(
                    ifd_model, args.model_path, i, args.identify
                )
            )

            with tqdm(total=len(futures), desc="Processing") as pbar:
                for future in futures:
                    pbar.update(1)

    run_all_evals(
        model_path=args.model_path,
        start_id=args.start_id + 1,
        end_id=args.end_id + 1,
        identify=args.identify,
        eval_data=args.eval_data,
        eval_to_path=args.eval_to_path,
        use_sample=args.use_sample,
        worker_number=args.worker_number,
    )


if __name__ == "__main__":
    main()
