"""Inference for FastChat models."""
import abc
import gc
import math
from typing import Optional
import sys
import random
import warnings
import argparse
import psutil
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
import json
import re

from fastchat.train.instruct_tuning import LazySupervisedDatasetVQA, LazySupervisedDatasetNoCaps, \
    LazySupervisedDatasetLLaVA_test

from fastchat.model.diff_llama import DiffLlamaForCausalLM

from fastchat.conversation import (
    conv_templates,
    SeparatorStyle,
)

from peft import PeftModel


def get_rand_des():
    text = ['Describe the image concisely.',
            'Provide a brief description of the given image.',
            'Offer a succinct explanation of the picture presented.',
            'Can you describe this image in short?',
            'Summarize the visual content of the image.',
            'Give a short and clear explanation of the subsequent image.',
            'Share a concise interpretation of the image provided.',
            'Present a compact description of the photo’s key features.',
            'Relay a brief, clear account of the picture shown.',
            'Render a clear and concise summary of the photo.',
            'Write a terse but informative summary of the picture.',
            'Create a compact narrative representing the image presented.']

    return text[random.randint(0, 11)]


def raise_warning_for_incompatible_cpu_offloading_configuration(device: str, load_8bit: bool, cpu_offloading: bool):
    if cpu_offloading:
        if not load_8bit:
            warnings.warn("The cpu-offloading feature can only be used while also using 8-bit-quantization.\n"
                          "Use '--load-8bit' to enable 8-bit-quantization\n"
                          "Continuing without cpu-offloading enabled\n")
            return False
        if not "linux" in sys.platform:
            warnings.warn(
                "CPU-offloading is only supported on linux-systems due to the limited compatability with the bitsandbytes-package\n"
                "Continuing without cpu-offloading enabled\n")
            return False
        if device != "cuda":
            warnings.warn("CPU-offloading is only enabled when using CUDA-devices\n"
                          "Continuing without cpu-offloading enabled\n")
            return False
    return cpu_offloading


def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def raise_warning_for_old_weights(model_path, model):
    if "vicuna" in model_path.lower() and isinstance(model, LlamaForCausalLM):
        if model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fastchat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.1: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template conv_one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n"
            )


def load_model(
        model_path, device, num_gpus, max_gpu_memory=None, load_8bit=False, cpu_offloading=False, debug=False
):
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(device, load_8bit, cpu_offloading)
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        print("init_kwargs", kwargs)
    else:
        raise ValueError(f"Invalid device: {device}")

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig
        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = str(math.floor(psutil.virtual_memory().available / 2 ** 20)) + 'Mib'
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=cpu_offloading)
        kwargs["load_in_8bit"] = load_8bit

    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    elif "dolly" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
    elif "pythia" in model_path or "stablelm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    elif "t5" in model_path:
        print("loading T5")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path,
                                                      low_cpu_mem_usage=True, **kwargs)
        tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False)
    elif "RWKV-4" in model_path:
        from fastchat.serve.rwkv_model import RwkvModel
        model = RwkvModel(model_path)
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m', use_fast=True)
    elif "diffllm" in model_path:
        model = DiffLlamaForCausalLM()
        model.load_state_dict(torch.load('/home/data2/xiangyu/Code/FastChat/output_diffllm'))
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        raise_warning_for_old_weights(model_path, model)
    if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


@torch.inference_mode()
def generate_stream(
        model, tokenizer, params, image, clip_image=None, device=None, context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = params.get("echo", True)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    input_ids = params["input_id"].to(device)
    # input_ids = tokenizer(prompt).input_ids

    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(input_ids=torch.as_tensor([input_ids],
                                                                 device=device))[0]

        start_ids = torch.as_tensor([[model.generation_config.decoder_start_token_id]],
                                    dtype=torch.int64, device=device)

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(input_ids=start_ids,
                                    encoder_hidden_states=encoder_output,
                                    use_cache=True)
                logits = model.lm_head(out[0])
            else:
                ask = tokenizer.decode(input_ids, skip_special_tokens=True,
                                       spaces_between_special_tokens=False)
                # out = model(input_ids=input_ids.unsqueeze(0), images=image, clip_l=clip_image, use_cache=True)
                out = model(input_ids=input_ids.unsqueeze(0), images=image, use_cache=True)
                # out = model(input_ids=torch.as_tensor([input_ids], device=device), images=image, use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(input_ids=torch.as_tensor([[token]], device=device),
                                    encoder_hidden_states=encoder_output,
                                    use_cache=True,
                                    past_key_values=past_key_values)

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(tmp_output_ids, skip_special_tokens=True,
                                      spaces_between_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, rfind_start)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def encode_stream(
        model, tokenizer, params, inputs_embedding, ids, attention_mask, device, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = params.get("echo", True)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)
    input_ids = ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)
    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(inputs_embeds=inputs_embedding)[0]

        start_ids = torch.as_tensor([[model.generation_config.decoder_start_token_id]],
                                    dtype=torch.int64, device=device)

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(input_ids=start_ids,
                                    encoder_hidden_states=encoder_output,
                                    use_cache=True)
                logits = model.lm_head(out[0])
            else:
                out = model(inputs_embeds=inputs_embedding, use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(input_ids=torch.as_tensor([[token]], device=device),
                                    encoder_hidden_states=encoder_output,
                                    use_cache=True,
                                    past_key_values=past_key_values)

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(tmp_output_ids, skip_special_tokens=True,
                                      spaces_between_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, rfind_start)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def image_path_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


def chat_loop(
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        load_8bit: bool,
        cpu_offloading: bool,
        conv_template: Optional[str],
        temperature: float,
        max_new_tokens: int,
        chatio: ChatIO,
        debug: bool,
):
    # Model
    # model, tokenizer = load_model(
    #     model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading, debug
    # )

    conv_template = "vicuna_v1.1"
    model = DiffLlamaForCausalLM.from_pretrained(
        "/home/data2/xiangyu/Code/FastChat/Tuning_for_LLaVA_MLP",
        cache_dir=None,
    )
    model.to('cuda')
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/home/data2/xiangyu/Code/FastChat/output_diffllm",
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    is_chatglm = "chatglm" in str(type(model)).lower()
    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_conv_template("one_shot").copy()

    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            generate_stream_func = generate_stream
            prompt = conv.messages[conv.offset:]
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        chatio.prompt_for_output(conv.roles[1])
        clip = None
        output_stream = generate_stream_func(model, tokenizer, gen_params, clip, device)
        outputs = chatio.stream_output(output_stream)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/fastchat-t5-3b-v1.0",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda",
        help="The device type"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="A single GPU like 1 or multiple GPUs like 0,2"
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading", action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU"
    )


def encoder2decoder(
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        load_8bit: bool,
        cpu_offloading: bool,
        conv_template: Optional[str],
        temperature: float,
        max_new_tokens: int,
        chatio: ChatIO,
        debug: bool,
):
    # Model
    # model, tokenizer = load_model(
    #     model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading, debug
    # )
    conv_template = "vicuna_v1.1"
    model = DiffLlamaForCausalLM.from_pretrained(
        "/home/data2/xiangyu/Code/EasyGen/Tuning_for_LLaVA_only_MLP",
        cache_dir=None,
    )
    model.to('cuda')
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/home/data2/xiangyu/Code/EasyGen/Tuning_for_LLaVA_MLP",
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    is_chatglm = "chatglm" in str(type(model)).lower()
    # data = LazySupervisedDatasetNoCaps(tokenizer=tokenizer)
    data = LazySupervisedDatasetLLaVA_test(tokenizer=tokenizer)
    length = data.__len__()
    f1 = open("/home/data2/xiangyu/Data/Diff-LLM/okvqa_vicuna.json", "a")
    f2 = open("/home/data2/xiangyu/Data/Diff-LLM/ka_truth.txt", "a")
    f3 = open("/home/data2/xiangyu/Data/Diff-LLM/ka_diff.txt", "a")
    result = []
    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = conv_templates("one_shot").copy()

    from tqdm import tqdm

    for i in tqdm(range(42226, length)):

        tmp = data.__getitem__(i)["images"]
        clip = tuple([torch.as_tensor(tmp, device='cuda')])
        clip = torch.stack(clip, 0)
        ids = data.__getitem__(i)["input_ids"][0][1:]
        image_id = data.__getitem__(i)["image_id"]
        # clip_image = data.__getitem__(i)["clip_l"]
        # clip_image = tuple([torch.as_tensor(clip_image, device='cuda')])
        # clip_image = torch.stack(clip_image, 0)
        # clip_image = clip_image.to(torch.float32)
        clip_image = None
        answer = data.__getitem__(i)["labels"]
        ask = tokenizer.decode(ids, skip_special_tokens=True,
                               spaces_between_special_tokens=False)

        conv.append_message(conv.roles[0], ask)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            generate_stream_func = encode_stream
            prompt = conv.messages[conv.offset:]
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        # encoder_output = torch.stack(llm, 0)
        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "input_id": ids,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        # chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, gen_params, clip, clip_image, device)
        outputs = chatio.stream_output(output_stream)
        # f1.write(outputs)
        # f1.write("\n")
        # f2.write(tokenizer.decode(ids, skip_special_tokens=True,
        #                               spaces_between_special_tokens=False))
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()
        # result.append({'question_id': str(image_id), 'answer': outputs})
        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    json_data_dict = json.dumps(result)
    f1.write(json_data_dict)


from fastchat.bidiffuser.sample_multi_v1 import get_image_feature, get_image

from absl import flags
from absl import app
from ml_collections import config_flags
import os
import sys
from fastchat.bidiffuser import utils
from fastchat.bidiffuser.libs.caption_decoder import CaptionDecoder
import fastchat.bidiffuser.libs.autoencoder
import clip


def inference_turn(
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        load_8bit: bool,
        cpu_offloading: bool,
        conv_template: Optional[str],
        temperature: float,
        max_new_tokens: int,
        chatio: ChatIO,
        debug: bool,
):
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config0", "fastchat/bidiffuser/configs/sample_unidiffuser_v1.py", "Configuration.", lock_config=False)
    FLAGS(sys.argv)
    config0 = FLAGS.config0
    nnet = utils.get_nnet(**config0.nnet)
    nnet.load_state_dict(
        torch.load('/home/data2/xiangyu/Code/EasyGen/fastchat/bidiffuser/models/uvit_v1.pth', map_location='cpu'))
    nnet.to(device)
    nnet.eval()

    caption_decoder = CaptionDecoder(device=device, **config0.caption_decoder)

    clip_text_model = fastchat.bidiffuser.libs.clip.FrozenCLIPEmbedder(device=device)
    clip_text_model.eval()
    clip_text_model.to(device)

    autoencoder = fastchat.bidiffuser.libs.autoencoder.get_model(**config0.autoencoder)
    autoencoder.to(device)

    clip_img_model, clip_img_model_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Model
    # model, tokenizer = load_model(
    #     model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading, debug
    # )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conv_template = "vicuna_v1.1"
    model = DiffLlamaForCausalLM.from_pretrained(
        "/home/data2/xiangyu/Code/EasyGen/pretrain_only_MLP",
        cache_dir=None,
    )

    model = PeftModel.from_pretrained(
        model,
        "/home/data2/xiangyu/Code/EasyGen/instruction_tunning_lora"
    )
    model.to(device)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/home/data2/xiangyu/Code/EasyGen/pretrain_only_MLP",
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )

    # data = LazySupervisedDatasetNoCaps(tokenizer=tokenizer)
    result = []
    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = conv_templates("one_shot").copy()
    image_path = chatio.image_path_for_input(conv.roles[0])
    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        generate_stream_func = generate_stream
        prompt = conv.get_prompt()[6:]
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0][1:]
        if image_path != '':
            image_feature = get_image_feature(image_path,
                                              nnet,
                                              caption_decoder,
                                              clip_text_model,
                                              autoencoder,
                                              clip_img_model,
                                              clip_img_model_preprocess)
        else:
            image_feature = None

        # tmp = data.__getitem__(i)["images"]
        # clip = tuple([torch.as_tensor(tmp, device='cuda')])
        # clip = torch.stack(clip, 0)
        # ids = data.__getitem__(i)["input_ids"][0][1:]
        # image_id = data.__getitem__(i)["image_id"]
        # clip_image = data.__getitem__(i)["clip_l"]
        # clip_image = tuple([torch.as_tensor(clip_image, device='cuda')])
        # clip_image = torch.stack(clip_image, 0)
        # clip_image = clip_image.to(torch.float32)

        # encoder_output = torch.stack(llm, 0)
        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "input_id": ids,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        # chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, gen_params, image_feature, None, device)
        outputs = chatio.stream_output(output_stream)
        des = re.findall(r"<Img>(.+?)</Img>", outputs)
        if des:
            get_image_feature(des,
                              nnet,
                              caption_decoder,
                              clip_text_model,
                              autoencoder,
                              clip_img_model,
                              clip_img_model_preprocess)
        # f1.write(outputs)
        # f1.write("\n")
        # f2.write(tokenizer.decode(ids, skip_special_tokens=True,
        #                               spaces_between_special_tokens=False))
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()
        # result.append({'question_id': str(image_id), 'answer': outputs})
        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


class SimpleChatIO(ChatIO):
    def image_path_for_input(self, role) -> str:
        return input(f"{'input image path or enter'}: ")

    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            outputs = outputs.strip().split(" ")
            now = len(outputs) - 1
            if now > pre:
                print(" ".join(outputs[pre:now]), end=" ")
                pre = now
        print(" ".join(outputs[pre:]))
        return " ".join(outputs)


def main(args):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.style == "simple":
        chatio = SimpleChatIO()

    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        inference_turn(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.temperature,
            args.max_new_tokens,
            chatio,
            args.debug,
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        help="Display style.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    args = parser.parse_args()
    main(args)
