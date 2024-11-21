# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import glob
import os
import random
import copy

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE_INDEX = LabelSmoother.ignore_index
DEFAULT_IMAGE_TOKEN = "<image>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    freeze_backbone: bool = field(default=False)
    tune_mlp: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    freeze_mlp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def get_feature_test(root):
    files = glob.glob(os.path.join(root, '*_tmp.npy'))
    num_data = len(files)
    return num_data


def get_rand_des():
    text = ['Describe the image concisely.',
            'Provide a brief description of the given image.',
            'Offer a succinct explanation of the picture presented.',
            'Can you describe this image briefly?',
            'Summarize the visual content of the image.',
            'Give a short and clear explanation of the subsequent image.',
            'Share a concise interpretation of the image provided.',
            'Present a compact description of the photo’s key features.',
            'Relay a brief, clear account of the picture shown.',
            'Render a clear and concise summary of the photo.',
            'Write a terse but informative summary of the picture.',
            'Create a compact narrative representing the image presented.']

    return text[random.randint(0, 11)]


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


# def make_supervised_data_module(
#     tokenizer: transformers.PreTrainedTokenizer, data_args
# ) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     dataset_cls = (
#         LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
#     )
#     rank0_print("Loading data...")
#
#     train_json = json.load(open(data_args.data_path, "r"))
#     train_dataset = dataset_cls(train_json, tokenizer=tokenizer)
#
#     if data_args.eval_data_path:
#         eval_json = json.load(open(data_args.eval_data_path, "r"))
#         eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
#     else:
#         eval_dataset = None
#
#     return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


class LazySupervisedDatasetVQA(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDatasetVQA, self).__init__()
        self.tokenizer = tokenizer
        self.root = "/home/data2/xiangyu/Data/coco512_features/vqa_diff"
        # self.root = "/home/data2/xiangyu/InstructTuning/Data/vqav2"
        self.num_data = get_feature_test(self.root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        # c = np.load(os.path.join(self.root, f'{head}_{tail}.npy'))
        # v = np.load(os.path.join(self.root, f'{head}_{tail}_llm.npy'))
        question = np.load(os.path.join(self.root, f'{index}_question.npy'))
        answer = np.load(os.path.join(self.root, f'{index}_answer.npy'))
        diff = np.load(os.path.join(self.root, f'{index}_tmp.npy'))
        diff = diff[0]
        clip = np.load(os.path.join(self.root, f'{index}_clip.npy'))
        clip = np.expand_dims(clip, axis=0)
        image_id = np.load(os.path.join(self.root, f'{index}_id.npy'))
        question = str(question).strip().replace('\n', '') + ' please giving an short answer.' + ' ASSISTANT:'
        query = get_rand_des()
        # question = query.strip().replace('\n', '') + ' ASSISTANT: '
        # answer = question + ' \n' + '### ASSISTANT: ' + str(answer[0]) + '\n'
        answer = str(answer)
        input_ids = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
        ).input_ids[0][1:]
        data_dict = dict(input_ids=input_ids, labels=answer, images=diff, image_id=image_id, clip=clip)
        return data_dict


class LazySupervisedDatasetNoCaps(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDatasetNoCaps, self).__init__()
        self.tokenizer = tokenizer
        self.root = "/home/data2/xiangyu/Data/coco512_features/kapathy_diffllm"
        self.num_data = get_feature_test(self.root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        text = np.load(os.path.join(self.root, f'{index}_text.npy'))
        llm_text = np.load(os.path.join(self.root, f'{index}_tmp.npy'))
        image_id = np.load(os.path.join(self.root, f'{index}_id.npy'))
        llm_text = llm_text[0]
        query = get_rand_des()
        text = str(text)
        answer = query.strip().replace('\n', '') + ' ASSISTANT: ' + text.strip().replace('\n', '') + '</s>'
        question = query.strip().replace('\n', '') + ' ASSISTANT:'
        instruction_len = len(self.tokenizer(query.strip().replace('\n', '')).input_ids)
        input_ids = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
        ).input_ids[0][1:]
        # target = input_ids.clone()
        # target[:instruction_len] = IGNORE_TOKEN_ID
        data_dict = dict(input_ids=input_ids, labels=text, images=llm_text, image_id=image_id)
        return data_dict


class LazySupervisedDatasetRandom(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDatasetRandom, self).__init__()
        self.tokenizer = tokenizer
        self.root = "/home/data2/xiangyu/Data/coco512_features/train_diff"
        # self.num_data = get_feature_test(self.root)
        self.num_data = 30000

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        head = index
        tail = random.randint(0, 3)
        # c = np.load(os.path.join(self.root, f'{head}_{tail}.npy'))
        # v = np.load(os.path.join(self.root, f'{head}_{tail}_llm.npy'))
        text = np.load(os.path.join(self.root, f'{head}_{tail}.npy'))
        llm_text = np.load(os.path.join(self.root, f'{head}_tmp.npy'))
        llm_text = llm_text[0]
        original_answer = np.load(os.path.join(self.root, f'{head}_{tail}.npy'))
        query = get_rand_des()
        text = str(text)
        # answer = query + ' \n' + '### ASSISTANT: ' + text.strip().replace('\n', '') + '\n'
        answer = query.strip().replace('\n', '') + ' ASSISTANT: ' + text.strip().replace('\n', '') + '</s>'
        instruction_len = len(self.tokenizer(query.strip().replace('\n', '') + ' ASSISTANT: ').input_ids)
        input_ids = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
        ).input_ids[0][1:]
        target = input_ids.clone()
        target[:instruction_len - 2] = IGNORE_TOKEN_ID
        data_dict = dict(input_ids=input_ids, labels=target, images=llm_text)
        return data_dict


class LazySupervisedDatasetQA(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.root = "/home/data2/xiangyu/Data/coco512_features/vqa_diff"
        self.num_data = get_feature_test(self.root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        # c = np.load(os.path.join(self.root, f'{head}_{tail}.npy'))
        # v = np.load(os.path.join(self.root, f'{head}_{tail}_llm.npy'))
        clip = np.load(os.path.join(self.root, f'{index}_clip.npy'))
        clip = np.expand_dims(clip, axis=0)
        question = np.load(os.path.join(self.root, f'{index}_question.npy'))
        answer = np.load(os.path.join(self.root, f'{index}_answer.npy'))
        diff = np.load(os.path.join(self.root, f'{index}_tmp.npy'))
        diff = diff[0]
        image_id = np.load(os.path.join(self.root, f'{index}_id.npy'))
        question = str(question).strip().replace('\n', '') + ' please giving an short answer.'
        # answer = question + ' \n' + '### ASSISTANT: ' + str(answer[0]) + '\n'
        answer = question + ' ASSISTANT: ' + str(answer[0]) + '</s>'
        instruction_len = len(self.tokenizer(question + ' ASSISTANT: ').input_ids)
        input_ids = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
        ).input_ids[0][1:]
        target = input_ids.clone()
        target[:instruction_len - 2] = IGNORE_TOKEN_ID
        data_dict = dict(input_ids=input_ids, labels=target, images=diff, clips=clip)
        return data_dict


class LazySupervisedDatasetVQAV2(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDatasetVQAV2, self).__init__()
        self.tokenizer = tokenizer
        self.root = "/home/data2/xiangyu/InstructTuning/Data/vqav2"
        self.num_data = get_feature_test(self.root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        # c = np.load(os.path.join(self.root, f'{head}_{tail}.npy'))
        # v = np.load(os.path.join(self.root, f'{head}_{tail}_llm.npy'))
        clip = np.load(os.path.join(self.root, f'{index}_clip.npy'))
        clip = np.expand_dims(clip, axis=0)
        question = np.load(os.path.join(self.root, f'{index}_conv.npy'))
        answer = np.load(os.path.join(self.root, f'{index}_answer.npy'))
        diff = np.load(os.path.join(self.root, f'{index}_tmp.npy'))
        diff = diff[0]
        image_id = np.load(os.path.join(self.root, f'{index}_id.npy'))
        question = str(question).strip().replace('\n', '') + ' please giving an short answer.'
        # answer = question + ' \n' + '### ASSISTANT: ' + str(answer[0]) + '\n'
        answer = question + ' ASSISTANT: ' + str(answer[0]) + '</s>'
        instruction_len = len(self.tokenizer(question + ' ASSISTANT: ').input_ids)
        input_ids = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
        ).input_ids[0][1:]
        target = input_ids.clone()
        target[:instruction_len - 2] = IGNORE_TOKEN_ID
        data_dict = dict(input_ids=input_ids, labels=target, images=diff, clips=clip)
        return data_dict


def preprocess_conv(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # Apply prompt templates
    sentence = ''
    conversations = []
    for source in sources:
        sentence += "USER: " + source['question'] + "? please giving an short answer. ASSISTANT: " + source[
            'answer'] + "</s>"
    conversations.append(sentence)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets
    sep = "ASSISTANT: "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split('</s>')
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids[0][4:],
        labels=targets[0][4:],
    )


def preprocess_llava(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # Apply prompt templates
    conversations = []
    if sources[0]["from"] != 'human':
        # Skip the first one if it is not from human
        sources = sources[1:]
    for source in sources:
        source["value"] = source["value"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        role = roles[source["from"]]
        conv.append_message(role, source["value"])
    conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets
    sep = "ASSISTANT: "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split('</s>')
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids[0][5:],
        labels=targets[0][5:],
    )


class LazySupervisedDatasetVisDial(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDatasetVisDial, self).__init__()
        self.tokenizer = tokenizer
        self.root = "/home/data2/xiangyu/Data/coco512_features/visdial_diffllm"
        self.num_data = get_feature_test(self.root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        # c = np.load(os.path.join(self.root, f'{head}_{tail}.npy'))
        # v = np.load(os.path.join(self.root, f'{head}_{tail}_llm.npy'))
        clip = np.load(os.path.join(self.root, f'{index}_clip.npy'))
        clip = np.expand_dims(clip, axis=0)
        diff = np.load(os.path.join(self.root, f'{index}_tmp.npy'))
        diff = diff[0]
        conversation = np.load(os.path.join(self.root, f'{index}_conv.npy'), allow_pickle=True)
        data_dict = preprocess_conv(conversation, self.tokenizer)
        data_dict['images'] = diff
        data_dict['clips'] = clip
        return data_dict


class LazySupervisedDatasetLLaVA(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDatasetLLaVA, self).__init__()
        self.tokenizer = tokenizer
        self.root = "/home/data2/xiangyu/InstructTuning/Data/LLaVA_80K"
        self.num_data = get_feature_test(self.root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        # c = np.load(os.path.join(self.root, f'{head}_{tail}.npy'))
        # v = np.load(os.path.join(self.root, f'{head}_{tail}_llm.npy'))
        clip = np.load(os.path.join(self.root, f'{index}_clip.npy'))
        clip = np.expand_dims(clip, axis=0)
        clip_l = np.load(os.path.join(self.root, f'{index}_clip-l.npy'))
        clip_l = clip_l[0]
        diff = np.load(os.path.join(self.root, f'{index}_tmp.npy'))
        diff = diff[0]
        conversation = np.load(os.path.join(self.root, f'{index}_conv.npy'), allow_pickle=True)
        data_dict = preprocess_llava(conversation, self.tokenizer)
        data_dict['images'] = diff
        data_dict['clips'] = clip
        data_dict['clip_l'] = clip_l
        return data_dict


class LazySupervisedDatasetLLaVA_test(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDatasetLLaVA_test, self).__init__()
        self.tokenizer = tokenizer
        self.root = "/home/data2/xiangyu/InstructTuning/Data/LLaVA_80K"
        self.num_data = get_feature_test(self.root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        # c = np.load(os.path.join(self.root, f'{head}_{tail}.npy'))
        # v = np.load(os.path.join(self.root, f'{head}_{tail}_llm.npy'))
        clip = np.load(os.path.join(self.root, f'{index}_clip.npy'))
        image_id = np.load(os.path.join(self.root, f'{index}_id.npy'))
        clip = np.expand_dims(clip, axis=0)
        clip_l = np.load(os.path.join(self.root, f'{index}_clip-l.npy'))
        clip_l = clip_l[0]
        diff = np.load(os.path.join(self.root, f'{index}_tmp.npy'))
        diff = diff[0]
        conversation = np.load(os.path.join(self.root, f'{index}_conv.npy'), allow_pickle=True)
        if conversation[0]["from"] != 'human':
            # Skip the first one if it is not from human
            conversation = conversation[1:]
        for source in conversation:
            source["value"] = source["value"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        input_ids = self.tokenizer(
            conversation[0]['value'],
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        labels = conversation[1]['value']
        data_dict = dict(input_ids=input_ids, labels=labels, images=diff, clips=clip, clip_l=clip_l, image_id=image_id)
        return data_dict


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # dataset_cls = SupervisedDataset
    # train_dataset = dataset_cls(tokenizer=tokenizer,
    #                             data_path=data_args.data_path,
    #                             preprocessed_path=data_args.preprocessed_path,
    #                             num_data=data_args.num_data)
    dataset_cls = LazySupervisedDatasetRandom
    train_dataset = dataset_cls(tokenizer=tokenizer)

    dataset_qa = LazySupervisedDatasetQA
    qa_dataset = dataset_qa(tokenizer=tokenizer)

    dataset_dialog = LazySupervisedDatasetVisDial
    dialog_dataset = dataset_dialog(tokenizer=tokenizer)

    dataset_llava = LazySupervisedDatasetLLaVA
    llava_dataset = dataset_llava(tokenizer=tokenizer)

    dataset_vqav2 = LazySupervisedDatasetVQAV2
    vqav2_dataset = dataset_vqav2(tokenizer=tokenizer)

    train_dataset = llava_dataset

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        clip = \
            tuple([torch.as_tensor(instance["clips"]) for instance in instances])
        clip = torch.stack(clip, 0)
        clip_l = \
            tuple([torch.as_tensor(instance["clip_l"]) for instance in instances])
        clip_l = torch.stack(clip_l, 0)
        images = \
            tuple([torch.as_tensor(instance["images"]) for instance in instances])
        images = torch.stack(images, 0)
        input_ids, target = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # labels = [instance["input_ids"] for instance in instances]
        # target = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        targets = torch.nn.utils.rnn.pad_sequence(target,
                                                  batch_first=True,
                                                  padding_value=IGNORE_TOKEN_ID)
        ret = dict(
            input_ids=input_ids,
            labels=targets,
            images=images,
            clip=clip,
            clip_l=clip_l,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        torch.set_printoptions(profile="full")
        return ret


from fastchat.model.diff_vicuna import DiffLlamaForCausalLM


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    # )
    model = DiffLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if model_args.tune_mlp:
        # model.requires_grad_(False)
        for p in model.get_model().fastchat_proj.parameters():
            p.requires_grad = True
        for p in model.get_model().clip_proj.parameters():
            p.requires_grad = True

    if training_args.freeze_mlp:
        for p in model.get_model().fastchat_proj.parameters():
            p.requires_grad = False
        for p in model.get_model().clip_proj.parameters():
            p.requires_grad = False

    params_grad = [n for n, p in model.named_parameters() if p.requires_grad]

    print(params_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Start training")
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
