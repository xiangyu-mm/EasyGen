from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM, \
    CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig


class DiffConfig(LlamaConfig):
    model_type = "diffllama"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiffLlamaModel(LlamaModel):
    config_class = DiffConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(DiffLlamaModel, self).__init__(config)

        self.fastchat_proj = Mlp(in_features=768, hidden_features=768 * 4, out_features=4096, act_layer=nn.GELU,
                                 drop=0.)

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path='/home/data2/xiangyu/Data/Vicuna-7b',
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

    def proj_image(self, tmp):
        device = tmp[-1].device
        inputs_fastchat = self.fastchat_proj(tmp)
        atts_image = torch.ones(inputs_fastchat.size()[:-1], dtype=torch.long).to(device)
        return inputs_fastchat, atts_image

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.tokenizer(
                p_before, return_tensors="pt", max_length=None).to(img_embeds.device)
            p_after_tokens = self.tokenizer(
                p_after, return_tensors="pt", max_length=None).to(img_embeds.device)
            # embed_tokens = self.fastchat.get_input_embeddings()
            p_before_embeds = self.embed_tokens(p_before_tokens.input_ids). \
                expand(batch_size, -1, -1)
            p_after_embeds = self.embed_tokens(p_after_tokens.input_ids[:, 1:]). \
                expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            # print(wrapped_img_embeds[0][60][:50])
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        inputs_embeds = self.embed_tokens(input_ids)

        if images is not None:
            device = images[-1].device
            img_embeds = self.fastchat_proj(images)
            atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(device)
            vqa_prompt = 'USER: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
            self.tokenizer.padding_side = "right"
            inputs_embeds = torch.cat([img_embeds, inputs_embeds], dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat([atts_img, attention_mask], dim=1)

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return super(DiffLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class DiffLlamaForCausalLM(LlamaForCausalLM):
    config_class = DiffConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = DiffLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path='/home/data2/xiangyu/Data/Vicuna-7b',
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def compute_len(self, prompt):
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.tokenizer(
            p_before, return_tensors="pt", max_length=None)
        p_after_tokens = self.tokenizer(
            p_after, return_tensors="pt", max_length=None)
        # embed_tokens = self.fastchat.get_input_embeddings()
        p_before_embeds = self.model.embed_tokens(p_before_tokens.input_ids). \
            expand(1, -1, -1).to('cuda')
        p_after_embeds = self.model.embed_tokens(p_after_tokens.input_ids[1:]). \
            expand(1, -1, -1).to('cuda')
        mask_len = p_before_embeds.shape[1]+77+p_after_embeds.shape[1]
        return mask_len

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(mask_len)
        if images is not None:
            empty_targets = (
                torch.ones([images.shape[0], 88],
                           dtype=torch.long).to('cuda').fill_(-100)  # plus one for bos
            )
        #
        if labels is not None:
            labels = torch.cat([empty_targets, labels], dim=1)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("diffllama", DiffConfig)
AutoModelForCausalLM.register(DiffConfig, DiffLlamaForCausalLM)
