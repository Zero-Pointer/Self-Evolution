# Self-Evolution

\[ [English](README.md) | 中文 \]

## 工件描述

该工件包含了Self-Evolution框架运行所需的全部程序代码以及样例数据，工件的整体结构以及文件的详细信息如下所示：

```
SELF-EVOLUTION       
│  draw_scores.py   // 实验结果柱状图绘制脚本
│  generation_main.py   // 主程序
│  LICENSE
│  main-abl-all.py   // 消融实验结果折线图绘制脚本
│  README.md
│  README_zh.md
│  requirements.txt // 程序运行需要的包
│
├─data  // 数据集
│      AIOps_data.json
│      AIOps_eval_data.json
│      toy_data.json
│      toy_eval_data.json
│
└─src
    │  export_model.py  // 模型合并脚本
    │  get_ifd_original.py  // IFD分数计算脚本
    │  self_evolution_generate.py   // 问答对生成脚本
    │  train_bash.py    // 模型训练脚本
    │
    └─llmtuner  // 指令调优工具
```

## 如何使用

### 环境准备

内存：`8GB`

显存：`32GB`

Python：`3.10.0`

CUDA Version：`12.1`

操作系统：`Ubuntu 24.04 LTS`

安装依赖： `pip install -r requirements.txt`

### 入门指南

在Self-Evolution项目目录下执行以下命令：

```shell
python generation_main.py \
    --start_id 0 \
    --end_id 3 \
    --source_data_file data/toy_data.json \
    --identify test \
    --model_path <你的初始模型的路径> \
    --eval_data data/toy_eval_data.json \
    --eval_to_path eval_result/qas_train \
    --mode hard \
    --train_epoch 1 \
    --ifd_k 5 \
    --batch_size 5 \
    --worker_number 1 \
    --infer_gpu 1 \
    --cuda_device 0 \
    --use_vllm
```

### 复现指南

在Self-Evolution项目目录下执行以下命令：

```shell
python generation_main.py \
    --start_id 0 \
    --end_id 8 \
    --source_data_file data/AIOps_data.json \
    --identify reproduce \
    --model_path <你的初始模型的路径> \
    --eval_data data/AIOps_eval_data.json \
    --eval_to_path eval_result/qas_train \
    --mode hard \
    --train_epoch 1 \
    --ifd_k 2000 \
    --batch_size 2000 \
    --worker_number 1 \
    --infer_gpu 1 \
    --cuda_device 0 \
    --use_vllm
```

程序运行结束后，在Self-Evolution项目目录下执行以下命令：

```shell
python draw_scores.py \
    --father_dir ./eval_result/qas_train/reproduce_generations \
    --res_dir exp_res/col_res
```

上述命令执行完毕后，即可在exp_res/col_res目录下得到根据本次实验结果绘制的柱状图。

在Self-Evolution项目目录下执行以下命令：

```shell
python main-abl-all.py \
    --father_dir ./eval_result/qas_train/reproduce_generations \
    --res_dir exp_res/abl_res
```

上述命令执行完毕后，即可在exp_res/abl_res目录下得到根据本次实验结果绘制的消融实验折线图。

### 参数说明

`--start_id`：起始轮数，默认为0，如果继续迭代，可以指定当前已迭代次数  
`--end_id`：结束迭代轮数，默认为8  
`--source_data_file`：原始知识片段文件路径  
`--identify`：此次工作流的独特标志符, 如果重复会覆盖先前生成的内容, 可以用数据量来标识  
`--eval_data`：评测数据集路径  
`--eval_to_path`：评测结果输出路径  
`--mode`：IFD筛选数据模式：

- `random` - 随机抽取
- `easy` - 简单题
- `even` - 均匀抽取
- `hard` - 困难题
- `all` - 所有数据都选，不筛选

`--train_epoch`：每一轮迭代训练的epoch  
`--ifd_k`：从历史数据中筛选出k条数据  
`--use_vllm`：使用vllm加速推理框架  
`--batch_size`：数据分块大小  
`--worker_number`：最大线程数  
`--infer_gpu`：推理阶段使用的gpu数量  
`--cuda_device`：模型训练阶段使用的gpu序号

## 数据格式

原始知识片段数据集样例：

```json
[
    {
        "instruction":"<原始知识片段>"
    },
    {
        "instruction":"<原始知识片段>"
    }
]
```

模型评测数据集样例：

```json
[
    {
        "instruction":"<原始知识片段>",
        "input":"<问题>",
        "output":"<预期答案>"
    },
    {
        "instruction":"<原始知识片段>",
        "input":"<问题>",
        "output":"<预期答案>"
    }
]
```

## 评测结果

```json
{
    "predict_bleu-4": 10.067049999999998,
    "predict_rouge-1": 32.988690000000005,
    "predict_rouge-2": 8.616989999999998,
    "predict_rouge-l": 20.33882,
    "predict_runtime": 65.3781,
    "predict_samples_per_second": 0.153,
    "predict_steps_per_second": 0.153
}
```

## 致谢

本项目受益于[Cherry_LLM](https://github.com/tianyi-lab/Cherry_LLM)和[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，项目样例数据来自[2024 CCF 国际AIOPS挑战赛数据集](https://www.modelscope.cn/datasets/issaccv/aiops2024-challenge-dataset)，感谢以上诸位作者的付出。
