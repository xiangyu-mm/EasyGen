import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from SUR_adapter import Adapter
import tempfile
from tools.fid_score import calculate_fid_given_paths
import torchvision.transforms as standard_transforms
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import numpy as np
from libs.caption_decoder import CaptionDecoder
import torch.nn as nn
import time
import torch.nn.functional as F
import random


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  # sample from q(xn|x0), where n is uniform
        if isinstance(x0, list):
            n = np.random.choice(list(range(1, self.N + 1)), (len(x0[0]),))
            eps = [torch.randn_like(tensor) for tensor in x0]
            xn = [stp(self.cum_alphas[n] ** 0.5, tensor) + stp(self.cum_betas[n] ** 0.5, _eps) for tensor, _eps in zip(x0, eps)]
            return torch.tensor(n), eps, xn
        else:
            # n = np.array(np.random.choice(list(range(1, self.N + 1)), (len(x0),)))
            n = np.array([random.randint(1, self.N)]*len(x0))
            eps = torch.randn_like(x0)
            xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return torch.tensor(n).to(x0.device), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'


def LSimple(x0, nnet, schedule, clip, z):
    n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
    t_img = torch.zeros(n.size(0), dtype=torch.int, device=n.device)
    data_type = torch.zeros_like(n, device=x0.device,
                                 dtype=torch.int) + 1
    z_out, clip_img_out, text_out = nnet(z, clip, text=xn, t_img=t_img, t_text=n, data_type=data_type)
    eps_z = torch.zeros(z_out.shape, device=n.device)
    # (self, img, clip_img, text, t_img, t_text, data_type)
    mse = nn.MSELoss()
    loss1 = mse(eps, text_out)
    loss2 = mse(z_out, eps_z)

    # n_re, eps_re, xn_re = schedule.sample(z)  # n in {1, ..., 1000}
    # t_txt = torch.zeros(n_re.size(0), dtype=torch.int, device=n.device)
    # data_type = torch.zeros_like(t_txt, device='cuda',
    #                              dtype=torch.int) + 1
    # z_out_re, clip_img_out_re, text_out_re = nnet(xn_re, clip, text=x0, t_img=n_re, t_text=t_txt, data_type=data_type)
    # eps_t = torch.zeros(text_out_re.shape, device=n.device)
    # loss3 = mse(eps_re, z_out_re)
    # loss4 = mse(text_out_re, eps_t)

    return loss1 + loss2


def LSimple1(x0, nnet, schedule, schedule1, clip, z):
    n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
    n_re, eps_re, xn_re = schedule1.sample(z)  # n in {1, ..., 1000}
    data_type = torch.zeros_like(n, device='cuda',
                                 dtype=torch.int) + 1
    z_out, clip_img_out, text_out = nnet(img=xn_re, clip_img=clip, text=xn, t_img=n_re, t_text=n, data_type=data_type)
    # eps_z = torch.zeros(z_out.shape, device=n.device)
    # (self, img, clip_img, text, t_img, t_text, data_type)
    mse = nn.MSELoss()
    loss1 = mse(eps, text_out)
    loss2 = mse(z_out, eps_re)

    # n_re, eps_re, xn_re = schedule.sample(z)  # n in {1, ..., 1000}
    # t_txt = torch.zeros(n_re.size(0), dtype=torch.int, device=n.device)
    # data_type = torch.zeros_like(t_txt, device='cuda',
    #                              dtype=torch.int) + 1
    # z_out_re, clip_img_out_re, text_out_re = nnet(xn_re, clip, text=x0, t_img=n_re, t_text=t_txt, data_type=data_type)
    # eps_t = torch.zeros(text_out_re.shape, device=n.device)
    # loss3 = mos(eps_re - z_out_re)
    # loss4 = mos(text_out_re - eps_t)
    print(loss1.detach().mean())
    print(loss2.detach().mean())
    # print(loss3)
    # print(loss4)
    return 20 * loss1 + loss2  # +loss3+loss4


def LSimple_T2I(img, clip_img, text, data_type, nnet, schedule, device, mask=None):
    r"""
    文到图loss
    """
    n, eps, xn = schedule.sample([img, clip_img])  # n in {1, ..., 1000}
    target, clip_img_eps = eps  # img_eps, clip_img_eps, target = eps
    img_eps, clip_img_eps = eps
    img_n, clip_img_n = xn
    n = n.to(device)
    clip_img_n = clip_img_n.to(torch.float32)
    t_text = torch.zeros_like(n, device=device)
    data_type = torch.zeros_like(t_text, device=device, dtype=torch.int) + 1
    img_out, clip_img_out, text_out = nnet(img_n, clip_img_n, text, t_img=n, t_text=t_text, data_type=data_type)

    # Compute instance loss
    aloss = F.mse_loss(img_out.float(), target.float(), reduction="mean")

    loss_img_clip = F.mse_loss(clip_img_out.float(), clip_img_eps.float(), reduction="mean")

    # n_text, eps_text, xn_text = schedule.sample(text)  # n in {1, ..., 1000}
    # n_text = n_text.to(device)
    # img_out1, clip_img_out1, text_out1 = nnet(img, clip_img, xn_text, t_img=t_text, t_text=n_text, data_type=data_type)
    # text_loss = F.mse_loss(text_out1.float(), eps_text.float(), reduction="mean")
    # # lora_img_out = torch.nn.functional.softplus(lora_img_out).mean()
    bloss = 1.2 * aloss + loss_img_clip

    return bloss


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    caption_decoder = CaptionDecoder(device=device, pretrained_path="models/caption_decoder.pth", hidden_dim=64)

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output_t2i.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True,
                                     drop_last=True,
                                     num_workers=8, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    print('loading model')
    nnet.load_state_dict(torch.load('/home/data2/xiangyu/Code/unidiffuser/models/uvit_v1.pth'))
    nnet_ema.load_state_dict(torch.load('/home/data2/xiangyu/Code/unidiffuser/models/uvit_v1.pth'))
    print('loading ok')
    lr_scheduler = train_state.lr_scheduler
    # train_state.resume(config.ckpt_root)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    adapter = Adapter(adapter_weight=0.2, sd_text_size=768)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                _, _context = data
                yield _context

    context_generator = get_context_generator()

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)
    _schedule = Schedule(_betas)
    _betas1 = stable_diffusion_beta_schedule()
    _schedule1 = Schedule(_betas1)
    logging.info(f'use {_schedule}')
    logging.info(f'use {_schedule1}')

    def cfg_nnet(x, timesteps, context):
        _cond = nnet_ema(x, timesteps, context=context)
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, timesteps, context=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond)

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        _z = autoencoder.sample(_batch[0])
        clip = _batch[1]
        text = _batch[2]
        # input llm representation and clip representation
        text = adapter(text, clip)
        contexts_low_dim = caption_decoder.encode_prefix(text)
        # t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)
        loss = LSimple(contexts_low_dim, nnet, _schedule, clip, _z)  # currently only support the extracted feature version
        # loss = LSimple_T2I(_z, clip, contexts_low_dim, None, nnet, _schedule, device)
        # _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        _metrics['loss'] = loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            return cfg_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)
        return decode(_z)

    def eval_step(n_samples, sample_steps, contexts):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        mode = 't2i'

        def split(x):
            C, H, W = config.z_shape
            z_dim = C * H * W
            z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
            z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
            clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
            return z, clip_img

        def combine(z, clip_img):
            z = einops.rearrange(z, 'B C H W -> B (C H W)')
            clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
            return torch.concat([z, clip_img], dim=-1)

        def unpreprocess(v):  # to B C H W and [0, 1]
            v = 0.5 * (v + 1.)
            v.clamp_(0., 1.)
            return v

        def sample_fn(mode, **kwargs):
            _n_samples = 1
            _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
            _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
            _text_init = torch.randn(_n_samples, 77, config.text_dim, device=device)
            if mode == 'joint':
                _x_init = combine_joint(_z_init, _clip_img_init, _text_init)
            elif mode in ['t2i', 'i']:
                _x_init = combine(_z_init, _clip_img_init)
            elif mode in ['i2t', 't']:
                _x_init = _text_init
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

            def model_fn(x, t_continuous):
                t = t_continuous * N
                if mode == 'joint':
                    return joint_nnet(x, t)
                elif mode == 't2i':
                    return t2i_nnet(x, t, **kwargs)
                elif mode == 'i2t':
                    return i2t_nnet(x, t, **kwargs)
                elif mode == 'i':
                    return i_nnet(x, t)
                elif mode == 't':
                    return t_nnet(x, t)

            dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
            with torch.no_grad():
                with torch.autocast(device_type='cuda'):
                    start_time = time.time()
                    x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
                    end_time = time.time()
                    print(
                        f'\ngenerate {_n_samples} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')

            os.makedirs('/home/data2/xiangyu/InstructTuning/Data/test_unidiffuser', exist_ok=True)
            if mode == 'joint':
                _z, _clip_img, _text = split_joint(x)
                return _z, _clip_img, _text
            elif mode in ['t2i', 'i']:
                _z, _clip_img = split(x)
                return _z, _clip_img
            elif mode in ['i2t', 't']:
                return x

        def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
            """
            1. calculate the conditional model output
            2. calculate unconditional model output
                config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
                config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
            3. return linear combination of conditional output and unconditional output
            """
            z, clip_img = split(x)

            t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

            z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                                 data_type=torch.zeros_like(t_text, device=device,
                                                                            dtype=torch.int) + config.data_type)
            x_out = combine(z_out, clip_img_out)

            if config.sample.scale == 0.:
                return x_out

            if config.sample.t2i_cfg_mode == 'empty_token':
                _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
                if use_caption_decoder:
                    _empty_context = caption_decoder.encode_prefix(_empty_context)
                z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=_empty_context,
                                                                          t_img=timesteps,
                                                                          t_text=t_text,
                                                                          data_type=torch.zeros_like(t_text,
                                                                                                     device=device,
                                                                                                     dtype=torch.int) + config.data_type)
                x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
            elif config.sample.t2i_cfg_mode == 'true_uncond':
                text_N = torch.randn_like(text)  # 3 other possible choices
                z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps,
                                                                          t_text=torch.ones_like(timesteps) * N,
                                                                          data_type=torch.zeros_like(t_text,
                                                                                                     device=device,
                                                                                                     dtype=torch.int) + config.data_type)
                x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
            else:
                raise NotImplementedError

            return x_out + config.sample.scale * (x_out - x_out_uncond)

        for idx, context in enumerate(contexts):
            contexts_low_dim = caption_decoder.encode_prefix(
                context.unsqueeze(0))
            _z, _clip_img = sample_fn(mode, text=contexts_low_dim)  # conditioned on the text embedding
            samples = unpreprocess(decode(_z))
            for idxx, sample in enumerate(samples):
                save_path = os.path.join('/home/data2/xiangyu/InstructTuning/Data/test_unidiffuser', 't2i', f'{idx}_{idxx}.png')
                save_image(sample, save_path)
            samples_pos = []
            for idxx, sample in enumerate(samples):
                sample_pil = standard_transforms.ToPILImage()(sample)
                sample_pil = utils.add_water(sample_pil)
                sample = standard_transforms.ToTensor()(sample_pil)
                samples_pos.append(sample)
            samples = make_grid(samples_pos, 1)
            save_path = os.path.join('/home/data2/xiangyu/InstructTuning/Data/test_unidiffuser', 'grid', f'{idx}_grid.png')
            save_image(samples, save_path)
        with tempfile.TemporaryDirectory() as temp_path:
            path = '/home/data2/xiangyu/InstructTuning/Data/test_unidiffuser/t2i'
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)

            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths(('/home/data2/xiangyu/InstructTuning/Data/test_unidiffuser/gt_512', path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval_t2i.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            # logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(metrics)
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
            eval_step(10000, 50, contexts)
        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)


from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'only_fine'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 't2i_training')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
