import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--father_dir", type=str, default="")
    parser.add_argument("--res_dir", type=str, default="res")
    parser.add_argument("--o", type=str, default="")
    args = parser.parse_args()
    return args


args = parse_args()
if args.o == "":
    t_dir = args.father_dir
    if t_dir[-1] == "/":
        t_dir = t_dir[:-1]
    args.o = t_dir.split("/")[-1]


def list_files(directory):
    results = []
    for root, dirs, files in os.walk(directory):
        t = []
        for file in files:
            if file.endswith("all_results.json"):
                t.append(os.path.join(root, file))
        results.extend(t)
    return results


def check_and_create_directory(path):
    if os.path.exists(os.path.dirname(path)):
        return True
    else:
        try:
            os.makedirs(os.path.dirname(path))
            return True
        except OSError as e:
            return False


qwen_7B_score = 7.2652

data = {
    "baseline": [
        ["7B", qwen_7B_score],
        ["72B", 13.8998],
        ["GPT-3.5", 14.4825],
        ["7B-HQ", 17.8487],
    ],
    "No Retrieval": [
        qwen_7B_score,
        8.6196,
        11.2595,
        14.0114,
        14.6405,
        14.9396,
        16.9243,
        17.7015,
        15.385,
    ],
    "All Retrieval": [qwen_7B_score, 8.8883, 12.1537, 17.3786, 19.2920, 18.0955],
    "Random Retrieval": [
        qwen_7B_score,
        8.7525,
        12.4203,
        14.1486,
        17.0797,
        18.5231,
        18.5418,
        16.9619,
        17.2573,
    ],
    "Self-Evolution": [qwen_7B_score],
}


if __name__ == "__main__":
    files = list_files(args.father_dir)

    for file in tqdm(files):
        with open(file, "r") as f:
            result = json.load(f)
            data["Self-Evolution"].append(result["predict_bleu-4"])

    titles = ["No Retrieval", "Random Retrieval", "All Retrieval"]
    ab_colors = ["orange", "purple", "green"]
    markers = ["o", "s", "^", "D", "p"]

    colors = ["green", "blue", "cyan", "magenta", "red"]
    linestyles = [
        (0, (3, 5, 1, 5)),
        "-.",
        ":",
        "--",
        "-",
    ]  # 普通线、虚线、点划线、点线、自定义线型

    for i, (bl, score) in enumerate(data["baseline"]):
        plt.hlines(
            score / 17.8487,
            -1,
            len(data[titles[0]]) + 2,
            label=bl,
            linestyle=linestyles[i],
        )

    data["Self-Evolution"] = [i / 17.8487 for i in data["Self-Evolution"][:9]]

    plt.plot(
        [i for i in range(0, len(data["Self-Evolution"]))],
        data["Self-Evolution"],
        marker="D",
        color="red",
        linestyle="-",
        label="Self-Evolution",
    )

    for i, title in enumerate(titles):
        data[title] = [i / 17.8487 for i in data[title][:9]]
        plt.plot(
            [i for i in range(0, len(data[title]))],
            data[title],
            marker=markers[i],
            color=ab_colors[i],
            linestyle="-",
            label=title,
        )

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.yticks([0.3, 1, 1.2])
    plt.xlim(-1, 9)
    plt.legend(loc="lower right")
    check_and_create_directory(os.path.join(args.res_dir, f"{args.o}.png"))
    plt.savefig(os.path.join(args.res_dir, f"{args.o}.png"), format="png", dpi=400)
    print("Figure saved:", os.path.join(args.res_dir, f"{args.o}.png"))
