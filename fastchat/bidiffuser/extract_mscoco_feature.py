import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm
import clip
from serve.inference import get_language_model, encode_stream
from PIL import Image


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='val')
    args = parser.parse_args()
    print(args)


    if args.split == "train":
        datas = MSCOCODatabase(root='/home/data2/xiangyu/Data/train2014',
                             annFile='/home/data2/xiangyu/Data/annotations/captions_train2014.json',
                             size=resolution)
        save_dir = f'/home/data2/xiangyu/Data/coco{resolution}_features/train_unidiffuer'
    elif args.split == "val":
        datas = MSCOCODatabase(root='/home/data2/xiangyu/Data/val2014',
                             annFile='/home/data2/xiangyu/Data/annotations/captions_val2014.json',
                             size=resolution)
        save_dir = f'/home/data2/xiangyu/Data/coco512_features/val_unidiffuer1'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    # os.makedirs(save_dir)

    autoencoder = libs.autoencoder.get_model('/home/data2/xiangyu/Code/unidiffuser/models/autoencoder_kl.pth')
    autoencoder.to(device)
    clip_img_model, clip_img_model_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_text = libs.clip.FrozenCLIPEmbedder()
    clip_text.eval()
    clip_text.to(device)

    model, tokenizer = get_language_model()
    #text = "Human: giving me some examples about renewable source. \n### Assistant:"
    #text_out = inference.encode_stream(model, tokenizer, text, context_len=2048, stream_interval=2)
    #text_out = process_text(text, only_encode= False)
    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            x, captions, image_pre = data

            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            # clip_img_feature = clip_img_model.encode_image(
            #     clip_img_model_preprocess(Image.fromarray(image_pre)).unsqueeze(0).to(device))
            # clip_img_feature = clip_img_feature.detach().cpu().numpy()
            # np.save(os.path.join(save_dir, f'{idx}_img.npy'), clip_img_feature)
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)
            #
            # latent = clip_text.encode(captions)
            # for i in range(len(latent)):
            #     c = latent[i].detach().cpu().numpy()
            #     np.save(os.path.join(save_dir, f'{idx}_{0}.npy'), c)


def process_text(text, model, tokenizer, only_encode=True):
    #text1 = "Human: giving me some examples about renewable source. \n### Assistant:"
    input_ids = tokenizer(text, padding="max_length", max_length=77, ).input_ids
    input_ids = input_ids[:77]
    encoder_output = model.encoder(input_ids=torch.as_tensor([input_ids],
                                                             device="cuda"))[0]
    if only_encode:
        return encoder_output

    stop_token_ids = []
    stop_token_ids.append(tokenizer.eos_token_id)
    prompt = text
    len_prompt = len(prompt)
    temperature = 0.7
    max_new_tokens = 256
    stop_str = "###"
    echo = False
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)
    context_len = 2048
    output = ""
    stream_interval = 2

    start_ids = torch.as_tensor([[model.generation_config.decoder_start_token_id]],
                                dtype=torch.int64, device="cuda")

    for i in range(max_new_tokens):
        if i == 0:
            out = model.decoder(input_ids=start_ids,
                                encoder_hidden_states=encoder_output,
                                use_cache=True)
            logits = model.lm_head(out[0])
            past_key_values = out.past_key_values
        else:
            out = model.decoder(input_ids=torch.as_tensor([[token]], device="cuda"),
                                encoder_hidden_states=encoder_output,
                                use_cache=True,
                                past_key_values=past_key_values)

            logits = model.lm_head(out[0])
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

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
            #yield output

        if stopped:
            break
    return output


if __name__ == '__main__':
    main()
