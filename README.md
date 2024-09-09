# Self-Evolution

\[ English | [中文](README_zh.md) \]

## Artifact Description

This artifact contains all the program code and sample data required for running the Self-Evolution framework. The overall structure of the artifact and detailed information about the files are as follows:

```
SELF-EVOLUTION       
│  draw_scores.py   // Script for drawing bar charts of experimental results
│  generation_main.py   // Main program
│  LICENSE
│  README.md
│  README_zh.md
│  requirements.txt // Packages required for program execution
│
├─data  // Datasets
│      AIOps_data.json
│      AIOps_eval_data.json
│      toy_data.json
│      toy_eval_data.json
│
└─src
    │  export_model.py  // Model Merge Script
    │  get_ifd_original.py  // IFD score calculation script
    │  self_evolution_generate.py   // Q&A generation script
    │  train_bash.py    // Model Training Script
    │
    └─llmtuner  // Instruction tuning tool
```

## How to Use

### Environment Setup

RAM:`8GB`

VRAM:`32GB`

Python:`3.10.0`

CUDA Version:`12.1`

OS:`Ubuntu 24.04 LTS`

Install dependencies:`pip install -r requirements.txt`

### Getting Started

Execute the following command in the Self-Evolution project directory:

```shell
python generation_main.py \
    --start_id 0 \
    --end_id 3 \
    --source_data_file data/toy_data.json \
    --identify test \
    --model_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
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

### Reproducibility Instructions

Execute the following command in the Self-Evolution project directory:

```shell
python generation_main.py \
    --start_id 0 \
    --end_id 8 \
    --source_data_file data/AIOps_data.json \
    --identify reproduce \
    --model_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
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

After the program runs, execute the following command in the Self-Evolution project directory:

```shell
python draw_scores.py \
    --father_dir ./eval_result/qas_train/reproduce_generations
```

After the above command is executed, the bar chart drawn based on the results of this experiment can be obtained in the current directory.

### Parameter Description

`--start_id`: Start iteration number, defaults to 0. If continuing iteration, specify the current number of iterations  
`--end_id`: End iteration number, defaults to 8  
`--source_data_file`: Path to the file containing the original knowledge fragments  
`--identify`: A unique identifier for this workflow; if repeated, it will overwrite the previously generated content and can be identified by the amount of data  
`--eval_data`: Path to the evaluation dataset  
`--eval_to_path`: Path to output the evaluation results  
`--mode`: IFD data filtering mode:

- `random` - Random selection
- `easy` - Easy questions
- `even` - Even extraction
- `hard` - Difficult questions
- `all` - Select all data without filtering

`--train_epoch`: The number of epochs for training in each iteration  
`--ifd_k`: Filter out k pieces of data from historical data  
`--use_vllm`: Use the vllm framework to accelerate inference  
`--batch_size`: The size of data chunks  
`--worker_number`: The maximum number of threads  
`--infer_gpu 1`: The number of GPUs used during the inference phase  
`--cuda_device`: The GPU identifier used during the model training phase  

## Data Format

Example of the original knowledge fragment dataset:

```json
[
    {
        "instruction":"<Original Knowledge Fragment>"
    },
    {
        "instruction":"<Original Knowledge Fragment>"
    }
]
```

Example of the model evaluation dataset:

```json
[
    {
        "instruction":"<Original Knowledge Fragment>",
        "input":"<Question>",
        "output":"<Expected Answer>"
    },
    {
        "instruction":"<Original Knowledge Fragment>",
        "input":"<Question>",
        "output":"<Expected Answer>"
    }
]
```

## Evaluation Results

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

## Acknowledgements

This project benefits from [Cherry_LLM](https://github.com/tianyi-lab/Cherry_LLM) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), the project sample data comes from the [2024 CCF International AIOPS Challenge dataset](https://www.modelscope.cn/datasets/issaccv/aiops2024-challenge-dataset), thanks to all the authors for their contributions.
