import logging
import os
import torch
import datasets
import transformers

from typing import Union, List
from datasets import load_dataset, concatenate_datasets

IGNORE_INDEX = -100

logger = logging.getLogger('__name__')

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant, 你是一个乐于助人的智能助手。"""
system_format = '<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
assistant_format = '{content}<|eot_id|>'


def build_instruction_dataset(
        data_path,
        tokenizer,
        max_seq_length,
        data_cache_dir=None,
        preprocessing_num_workers=None,
        ):
        def tokenization(examples):
            sources = []
            targets = []
            for instruction, input_txt, output in zip(examples['instruction'], examples['input'], examples['output']):
                if input_txt is not None and input_txt != "":
                    instruction = instruction + '\n' + input_txt

                sources.append(system_format.format(content=DEFAULT_SYSTEM_PROMPT) + user_format.format(content=instruction))
                targets.append(assistant_format.format(content=output))

            tokenized_src = tokenizer(sources, return_attention_mask=False, add_special_token=False)
            tokenized_tgt = tokenizer(targets, return_attention_mask=False, add_special_token=False)

            all_input_ids = []
            all_labels = []

            for s, t in zip(tokenized_src['input_ids'], tokenized_tgt['input_ids']):
                input_ids = torch.LongTensor(s + t)[:max_seq_length]
                labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]  # 问题部分不计算 loss, max_seq_length 处截断
                all_input_ids.append(input_ids)
                all_labels.append(labels)

            results = {'input_ids': all_input_ids, 'labels': all_labels}
            return results

        logging.warning('-' * 30 + ' Building dataset ' + '-' * 30)
        all_datasets = []

        if not isinstance(data_path, (list, tuple)):
            data_path = [data_path]
        
        for file in data_path:
            if data_cache_dir is None:
                data_cache_dir = str(os.path.dirname(file))

            cache_path = os.path.join(data_cache_dir, os.path.basename(file).split('.')[0] + f'_{max_seq_length}')
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path)
                logger.info(f'training datasets-{file} has been loaded from disk')
            except Exception:
                raw_dataset = load_dataset('json', data_files=file, cache_dir=cache_path)
                tokenization_func = tokenization
                tokenized_dataset = raw_dataset.map(
                    tokenization_func,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=['instruction', 'input', 'output'],
                    keep_in_memory=False,
                    desc='preprocessing on dataset',
                )
                processed_dataset = tokenized_dataset
                processed_dataset.save_to_disk(cache_path)
            
            processed_dataset.set_format('torch')
            all_datasets.append(processed_dataset['train'])
        
        all_datasets = concatenate_datasets(all_datasets)
        return all_datasets