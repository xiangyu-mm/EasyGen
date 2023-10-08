import json
import requests
import os
import time
from PIL import Image
import numpy as np
import einops
from io import BytesIO
from torch.utils.data import Dataset
import libs.autoencoder
import libs.clip
import tqdm
import torch
import clip


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


def extract_and_download_links(data, detail_data=None):
    # 如果数据是字典类型，遍历其键值对
    if detail_data is None:
        detail_data = []
    if isinstance(data, dict):
        des = ""
        ids = ""
        img = ""
        dialogue = []
        for key, value in data.items():
            mydict = {}
            # 如果值是字符串类型，并且以http或https开头，说明是一个链接
            if key == "photo_description":
                des = value

            elif key == "photo_id":
                ids = value.split('/')
                ids = ids[1]

            elif isinstance(value, str) and value.startswith(("http", "https")) and key == "photo_url":
                # 打印出链接
                img = value

            elif key == "dialogue":
                dialogue = value

            else:
                extract_and_download_links(value, detail_data)

            if ids != "":
                mydict['image'] = img
                mydict['caption'] = des
                mydict['dialogue'] = dialogue
                detail_data.append(mydict)

    # 如果数据是列表类型，遍历其元素
    elif isinstance(data, list):
        for element in data:
            # 递归调用函数处理元素
            extract_and_download_links(element, detail_data)

    return detail_data


class PhotoChatDatabase(Dataset):
    def __init__(self, data):
        self.data = data
        self.detail_data = extract_and_download_links(data)

    def __len__(self):
        return len(self.detail_data)

    def process_data(self, key: int):
        img, des = self.detail_data[key].get('image'), self.detail_data[key].get('caption')
        img = requests.get(img)
        image = img.content
        image = Image.open(BytesIO(image)).convert('RGB')
        image = np.array(image).astype(np.uint8)
        image = center_crop(256, 256, image)
        clip_img = image
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')
        dialogue = self.detail_data[key].get('dialogue')
        return clip_img, image, des, dialogue

    def __getitem__(self, index):
        try:
            image_clip, image, target, dialogue = self.process_data(index)
            return image_clip, image, target, dialogue
        except:
            image_clip = ''
            image = ''
            target = ''
            dialogue = ''
            return image_clip, image, target, dialogue


# 调用函数处理json数据

def main():
    file = r'data/photochat/dev/'
    data_sum = []
    for root, dirs, files in os.walk(file):
        if root != file:
            break
        for file in files:
            path = os.path.join(root, file)

            with open(path, 'r', encoding='utf-8') as fp:
                # load()函数将fp(一个支持.read()的文件类对象，包含一个JSON文档)反序列化为一个Python对象
                data = json.load(fp)
                data_sum = data_sum + data
    datas = PhotoChatDatabase(data_sum)
    # print(datas.__getitem__(1))
    save_dir = f'data/photochat_feature/dev'
    device = "cuda"
    os.makedirs(save_dir)

    autoencoder = libs.autoencoder.get_model('models/autoencoder_kl.pth')
    autoencoder.to(device)
    clip_text_model = libs.clip.FrozenCLIPEmbedder(device=device)
    clip_text_model.eval()
    clip_text_model.to(device)
    clip_img_model, clip_img_model_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    empty_context = clip_text_model.encode([''])[0]
    with torch.no_grad():
        for idx, data in enumerate(datas):
            clip_img, x, captions, dialogue = data
            if clip_img == '':
                continue
            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)

            clip_img_feature = clip_img_model.encode_image(
                clip_img_model_preprocess(Image.fromarray(clip_img)).unsqueeze(0).to(device))
            clip_img_feature = clip_img_feature.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}_clip.npy'), clip_img_feature)

            latent = clip_text_model.encode(captions)
            c = latent.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}_text.npy'), c)

            np.save(os.path.join(save_dir, f'{idx}_dialogue.npy'), dialogue)


if __name__ == '__main__':
    main()
