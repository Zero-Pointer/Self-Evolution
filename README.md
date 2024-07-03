# Self-Evolution

## 如何使用

### 环境准备

安装依赖： `pip install -r requirements.txt`

### 快速开始

示例：

```shell
python generation_main.py \
    --start_id 0 \
    --end_id 3 \
    --python_path <your_path_to_python> \
    --source_data_file data/source_data.json \
    --identify test \
    --model_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --eval_data data/eval_data.json \
    --eval_to_path eval_result/qas_train \
    --mode even \
    --train_epoch 1 \
    --ifd_k 5 \
    --batch_size 5 \
    --worker_number 1 \
    --infer_gpu 1 \
    --cuda_device 0 \
    --use_vllm
```

`--start_id`：起始轮数，默认为0，如果继续迭代，可以指定当前已迭代次数  
`--end_id`：结束迭代轮数，默认为8  
`--source_data_file`：原始知识片段文件路径  
`--identify`：此次工作流的独特标志符, 如果重复会覆盖先前生成的内容, 可以用数据量来标识  
`--eval_data`：评测数据集路径  
`--eval_to_path`：评测结果输出路径  
`--mode`：IFD筛选数据模式：

- `random` - 随机
- `easy` - 简单题
- `even` - 均匀抽取
- `hard` - 困难题
- `all` - 所有数据都选，不筛选

`--train_epoch`：每一轮迭代训练的epoch  
`--ifd_k`：从历史数据中筛选出k条数据  
`--use_vllm`：使用vllm加速推理框架  
`--batch_size`：数据分块大小  
`--worker_number`：最大线程数  
`--infer_gpu 1`：推理阶段使用的gpu数量  
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

本项目受益于[Cherry_LLM](https://github.com/tianyi-lab/Cherry_LLM)和[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，感谢以上诸位作者的付出。