# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling, Trainer

from ...data import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_custom_optimzer, create_modelcard_and_push


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    print("Ready for tokenizer")
    tokenizer = load_tokenizer(model_args)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="pt")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    optimizer = create_custom_optimzer(model, dataset, training_args, finetuning_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=(optimizer, None),
        **split_dataset(dataset, data_args, training_args),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        
#         import torch
        
#         for i in range(len(trainer.state.log_history)):
#             print(trainer.state.log_history[i])
#             if 'grad_norm' not in trainer.state.log_history[i].keys():
#                 continue
#             if isinstance(trainer.state.log_history[i]['grad_norm'], torch.Tensor):
#                 trainer.state.log_history[i]['grad_norm'] = trainer.state.log_history[i]['grad_norm'].item()
                
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
