# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py
import os

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq, AutoModelForCausalLM
from peft import PeftModel
import torch

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ...train.sft.metric import ComputeMetrics
from ...train.sft.trainer import CustomSeq2SeqTrainer
from ...train.utils import create_modelcard_and_push
from ..utils import create_custom_optimzer
from ...extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)
def print_nvidia():
    import subprocess
    # import pandas as pd

    # 构建命令
    command = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv"]

    # 执行命令
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 等待命令执行完成并获取输出
    stdout, stderr = process.communicate()

    print("NVIDIA-SMI Output (CSV):")
    print(stdout)
def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    print(f"GPU number: {torch.cuda.device_count()}")
    tokenizer = load_tokenizer(model_args)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
    print_nvidia()
#     exit()
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
#     model = AutoModelForCausalLM.from_pretrained('/root/share/zpt/projects/rag_sft/model/Qwen1.5-32B-chat', device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)

#     if training_args.do_predict:
#         # modify the output path
#         if training_args.output_dir == '':
#             adapter = model_args.adapter_name_or_path[0]
#             predict_file = data_args.dataset_file.split('/')[-1].split('.')[0]
#             training_args.output_dir = os.path.join(adapter, predict_file)    

    if training_args.do_predict:
        # modify the output path
        if model_args.adapter_name_or_path is None:
            adapter = 'generation_0'
        else:
            adapter = model_args.adapter_name_or_path[0].split('/')[-1]
        training_args.output_dir = os.path.join(training_args.output_dir, adapter)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    # Initialize our Trainer
    optimizer = create_custom_optimzer(model, dataset, training_args, finetuning_args)
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=(optimizer, None),
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    
#     print('-------------------------------------------')
#     for name, param in model.named_parameters():
#         print(f"Parameters: {name}, Data Type: {param.dtype}")
#     print('-------------------------------------------')

    # Training
    if training_args.do_train:
        print('-------------------------------------------')
        print(f"trainer.is_world_process_zero():{trainer.is_world_process_zero()}")
        print(f"trainer.is_local_process_zero():{trainer.is_local_process_zero()}")
        print('-------------------------------------------')
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        train_result.metrics['dataset_file'] = data_args.dataset_file
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        
#         vars_of_state = vars(trainer.state)
#         print('-------------------------------------------')
#         print("vars_of_state: ")
#         print(vars(trainer.state))
#         print(trainer.state.log_history)
#         print('-------------------------------------------')
        
#         import torch
        
#         for i in range(len(trainer.state.log_history)):
#             print(trainer.state.log_history[i])
#             if 'grad_norm' not in trainer.state.log_history[i].keys():
#                 continue
#             if isinstance(trainer.state.log_history[i]['grad_norm'], torch.Tensor):
#                 trainer.state.log_history[i]['grad_norm'] = trainer.state.log_history[i]['grad_norm'].item()
                
#         print('-------------------------------------------')
#         print("vars_of_state: ")
#         print(trainer.state.log_history)
#         print('-------------------------------------------')
        print('Saving state!')
        try:
            trainer.save_state()
        except e:
            print('-------------------------------------------')
            print(e)
            print('Save state fail again!')
            print(trainer.state)
            print('-------------------------------------------')
        print('Saving state finish!')
                
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
    if trainer.is_world_process_zero():
        logger.info("pt_program_ends")
    # Create model card
#     create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

def get_all_checkpoints(adapter):
    paths = []
    for p in os.listdir(adapter):
        if "checkpoint-" in p:
            paths.append([p, os.path.join(adapter, p)])
    return paths

def get_all_generations(adapter):
    paths = []
    for p in os.listdir(adapter):
        if "generation_" in p or "checkpoint-" in p:
            paths.append([p, os.path.join(adapter, p)])
    return paths

def run_sft_predict_all(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer = load_tokenizer(model_args)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
    output_dir = training_args.output_dir
    # Todo
    # get all checkpoints of adapter to adapters 
    # ckps = get_all_checkpoints(model_args.adapter_name_or_path[0])
    ckps = get_all_generations(model_args.adapter_name_or_path[0])
    adapter_name_or_path = model_args.adapter_name_or_path[0]
    model_args.adapter_name_or_path = None

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    for i in range(len(ckps)):
        adapter_name = ckps[i][0]
        adapter_path = ckps[i][1]
        if i == 0:
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
        else:
            model.load_adapter(adapter_path, adapter_name=adapter_name)
    # loop all checkpoints
    for ckp in ckps:
        adapter_name = ckp[0]
        adapter_path = ckp[1]
        model.set_adapter(adapter_name)
        print("---------------------------------------------------------------")
        print("Now Inferencing the {}".format(adapter_path))
        print("---------------------------------------------------------------")

        if training_args.do_predict:
            # modify the output path
            training_args.output_dir = os.path.join(output_dir, adapter_name_or_path.split('/')[-1], adapter_name)

        if training_args.predict_with_generate:
            tokenizer.padding_side = "left"  # use left-padding in generation

        if getattr(model, "is_quantized", False) and not training_args.do_train:
            setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        )

        # Override the decoding parameters of Seq2SeqTrainer
        training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
        training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

        # Initialize our Trainer
        optimizer = create_custom_optimzer(model, dataset, training_args, finetuning_args)
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            optimizers=(optimizer, None),
            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
            **split_dataset(dataset, data_args, training_args),
        )

        # Keyword arguments for `model.generate`
        gen_kwargs = generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()

        # Predict
        if training_args.do_predict:
            predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
            if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
                predict_results.metrics.pop("predict_loss", None)
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            trainer.save_predictions(predict_results)

        
        model = model.to('cpu')
        # 释放掉未使用的缓存
        import torch
        torch.cuda.empty_cache()
        # free the model location