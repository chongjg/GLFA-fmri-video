import yaml
import numpy as np
import torch
import os
import h5py
import cv2
import decord
decord.bridge.set_bridge('torch')
from einops import rearrange

import imageio
import random
import torchvision
from torchvision import transforms

from models.transformer_models import *

from diffusers.pipelines.text_to_video_synthesis import *
from diffusers.utils.import_utils import is_xformers_available
from typing import Any, Callable, Dict, List, Union, Optional

import open_clip

from timm.models.layers import Mlp, DropPath

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

from models.LocallyConnected2d import LocallyConnected2d

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

# data augmentation code in nlp from https://maelfabien.github.io/machinelearning/NLP_8/#
from nltk.corpus import wordnet 

def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
     
    for syn in wordnet.synsets(word):
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
   
    return list(synonyms)

def synonym_replacement(words, n):
    
    words = words.split()
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence

def swap_word(new_words):
    
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        
        if counter > 3:
            return new_words
    
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

def random_swap(words, n):
    
    words = words.split()
    new_words = words.copy()
    
    for _ in range(n):
        new_words = swap_word(new_words)
        
    sentence = ' '.join(new_words)
    
    return sentence

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=self.size)(img)
        return img

class fd_fmri_video_dataset(torch.utils.data.Dataset):

    def text_transform(self, words):
        words = random_swap(words, n=1)
        words = synonym_replacement(words, n=1)
        return words

    def mask_init(self):
        # Mask
        mask = np.load(self.path + 'vc_roi.npz')['images'][0] # [H, W]
        H, W = mask.shape
        vc_mask = mask == 1 # [H, W]
        fg_mask = (mask == 1) | (mask == -1) # [H, W]

        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        xx, yy = np.meshgrid(x, y)
        grid = np.stack([xx, yy], axis=0) # [2, H, W]

        gird_ = grid * vc_mask[np.newaxis]
        x1 = min(int(gird_[0].max()) + 1, W)
        y1 = min(int(gird_[1].max()) + 10, H)
        gird_[gird_ == 0] = 1e6
        x0 = max(int(gird_[0].min() - 1), 0)
        y0 = max(int(gird_[1].min() - 10), 0)
        self.vc_mask = vc_mask
        self.fg_mask = fg_mask
        self.coord = [x0, x1, y0, y1]
        self.crop_msk = self.vc_mask[self.coord[2]:self.coord[3] + 1, self.coord[0]:self.coord[1] + 1]
        self.cmask = cv2.resize(self.crop_msk * 1., (self.fmri_size[1], self.fmri_size[0]), interpolation=cv2.INTER_NEAREST)  # (W, H)

    def __init__(self,
                 path='./',
                 video_size=(576, 320),
                 fmri_size=(256, 256),
                 vid_per_run=50,
                 del_frames=3,
                 lag=5, # lag 4s (5*0.8s)
                 window_size=5,
                 phase='train',

                 n_sample_frames: int = 24,
                 sample_frame_rate: int = 3,

                 subjects = [1,2,3,4,5],
                 stage='fmri-video',

                 data_aug: bool = False,
        ):

        super().__init__()

        assert phase in ['train', 'test']
        
        self.path = path
        self.vid_per_run = vid_per_run
        self.stage = stage
        self.data_aug = data_aug
        if self.data_aug:
            print('using data augmentation.')
            self.image_transform = transforms.Compose([
                random_crop((round(video_size[1] * 0.8), round(video_size[0] * 0.8)), p=0.5),
                transforms.Resize(video_size),
            ])

        self.subjects = subjects
        if phase == 'train':
            self.fmri_run = [[2,3,5,6,7], [5,6,7,8,9,10,11,12,17,18,19,20,21,22,23,24,25,26,27,28]]
            self.len = 1000
        else:
            self.fmri_run = [[1,4], [1,2,3,4,13,14,15,16]]
            self.len = 400 // 2

        self.len *= 2 # 8s video -> 2 * 4s sample

        self.phase = phase
        
        print('loading ' + phase + ' dataset')

        #################### prompt #####################

        self.vid2prompt = np.load(os.path.join(path, 'stimuli/vid2name.npy'), allow_pickle=True).item()
        print('prompt loaded.')

        #################### video #####################

        self.width = video_size[0]
        self.height = video_size[1]
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate

        self.vids = []
        for runid in self.fmri_run[0]:
            with open(os.path.join(path, f'stimuli/condition/run{runid}.txt')) as f:
                vids = f.readlines()
                vids = [int(vid.replace('\n', '')) for vid in vids]
                self.vids.append(vids)
        self.vids = np.array(self.vids).reshape(-1, vid_per_run)

        if os.path.exists(os.path.join(path, f'stimuli/all-{phase}-video.npy')):
            self.video_data = np.load(os.path.join(path, f'stimuli/all-{phase}-video.npy'))
        else:
            self.video_data = []
            for runid in range(len(self.fmri_run[1])):
                self.video_data.append([])
                run_vids = self.vids[runid]
                print(f'video loading ... {runid}/{len(self.fmri_run[1])}')
                for vid in run_vids:
                    video = decord.VideoReader(os.path.join(path, f'stimuli/{vid}.mp4'), width=self.width, height=self.height)

                    sample_index = [i * 30 // sample_frame_rate for i in range(n_sample_frames)]
                    sample = video.get_batch(sample_index)
                    sample = rearrange(sample, "f h w c -> f c h w")
                    self.video_data[-1].append((sample.numpy() / 127.5 - 1.0).astype(np.float16))
                    del video
            self.video_data = np.array(self.video_data)
            np.save(os.path.join(path, f'stimuli/all-{phase}-video.npy'), self.video_data)

        print('video loaded.')

        #################### fmri #####################

        self.fmri_frames = 10 + window_size # 8s video + 5*0.8s lag
        self.fmri_size = fmri_size # (H, W)
        self.del_frames = del_frames
        self.lag = lag
        self.window_size = window_size

        self.mask_init()

        fmri_datapath = os.path.join(path, f'stimuli/all-{phase}-fmri.npy')
        if os.path.exists(fmri_datapath):
            self.fmri_data = np.load(fmri_datapath)
        else:
            self.fmri_data = []
            for subid in [1,2,3,4,5]:
                print(f'fmri loading ... {subid}/5')
                self.fmri_data.append([])
                for runid in self.fmri_run[subid>1]:
                    print(f'fmri loading run-{runid}')
                    # self.fmri_data[-1].append([])
                    file_path = os.path.join(self.path, f'processed-fmri/sub-000{subid}/hdf5/sub-000{subid}_task-video_run-{runid}_space-fsLR_den-91k_bold.dtseries.h5')
                    fmri_h5 = h5py.File(file_path, 'r')["images"][self.del_frames+self.lag:]
                    # print(fmri_h5.shape)
                    fmri_h5 = self.fmriNorm(fmri_h5)
                    print(f'fmri -- mean : {fmri_h5[fmri_h5!=0].mean()}, min : {fmri_h5.min()}, max : {fmri_h5.max()}')
                    tmp_fmri_list = []
                    cnt_lost = 0
                    for timid in range(vid_per_run * (4 if subid==1 else 1)*13+self.fmri_frames-13):
                        if timid >= fmri_h5.shape[0]:
                            print(f'sub{subid}--run{runid}--tim{timid}')
                            tmp_fmri_list.append(tmp_fmri_list[-1] * 0)
                            cnt_lost += 1
                        # 8s video + 2.4s blank = 13*0.8s 
                        else:
                            tmp_fmri_list.append(self.fmriPreprocess(fmri_h5[timid]))
                    print(f'lost {cnt_lost} frames')
                    if subid > 1:
                        self.fmri_data[-1].append(tmp_fmri_list)
                    else:
                        for tmpid in range(4):
                            self.fmri_data[-1].append(tmp_fmri_list[tmpid*vid_per_run*13 : (tmpid+1)*vid_per_run*13+self.fmri_frames-13])
                    del fmri_h5
                
            self.fmri_data = np.array(self.fmri_data)
            np.save(fmri_datapath, self.fmri_data)
        self.fmri_data = self.fmri_data[[sub-1 for sub in self.subjects]]
        if phase == 'test':
            self.fmri_data = self.fmri_data.reshape(self.fmri_data.shape[0],2,-1,self.fmri_data.shape[2],self.fmri_data.shape[3],self.fmri_data.shape[4]).mean(1)
        print('fmri load finished.')

        self.aver_fmri = torch.from_numpy(np.concatenate([self.fmri_data[:,:,i*13:i*13+self.window_size] for i in range(self.vid_per_run)] + [self.fmri_data[:,:,i*13+5:i*13+5+self.window_size] for i in range(self.vid_per_run)])).mean((0,1))[:,None]

    def __len__(self):
        return self.len

    def fmriNorm(self, fmri):
        fmri_mean = np.mean(fmri, 0, keepdims=True)
        fmri_std = np.std(fmri, 0, keepdims=True, ddof=1)
        fmri = ((fmri - fmri_mean) / fmri_std)
        fmri[np.isnan(fmri)] = 0
        return fmri

    def fmriPreprocess(self, image):
        image = np.array(image)
        image = image[self.coord[2]:self.coord[3] + 1, self.coord[0]:self.coord[1] + 1]
        image = cv2.resize(image, (self.fmri_size[1], self.fmri_size[0]))
        image[self.cmask == 0] = 0
        return image

    def getIndex(self, index):
        offset = (index % 2)
        index //= 2

        runid = index // self.vid_per_run
        vidid = index %  self.vid_per_run
        timid = vidid * 13
        return runid, vidid, timid, offset

    def __getitem__(self, index):
        runid, vidid, timid, offset = self.getIndex(index)
        # print(f'{runid} {vidid} {timid} {offset}')
        timid += offset * 5

        vid = self.vids[runid, vidid]

        text_prompt = self.vid2prompt[vid]

        video = self.video_data[runid, vidid, self.n_sample_frames//2*offset:self.n_sample_frames//2*offset+self.n_sample_frames//2]
        video = torch.from_numpy(video)
        
        if self.stage == 'fmri-text':
            frame_id = random.randint(0, self.n_sample_frames//2-1)
            video = (video[frame_id] + 1) / 2

        fmri = self.fmri_data[:,runid, timid : timid + self.window_size]
        fmri = torch.from_numpy(fmri)[:, :, None]

        if self.data_aug:
            video = self.image_transform(video)
            text_prompt = self.text_transform(text_prompt)

        item = {'text_prompt': text_prompt, 'pixel_values': video, 'image': fmri}
        return item

def torch_init_model(model, state_dict):
    # state_dict = torch.load(init_checkpoint, map_location='cpu')[key]
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    print("  --> missing keys:{}".format(missing_keys))
    print('  --> unexpected keys:{}'.format(unexpected_keys))
    print('  --> error msgs:{}'.format(error_msgs))

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, macro_block_size=None)
    # imageio.mimsave(path, outputs, duration=int(1000/fps))

class fMRI_encoder(torch.nn.Module):
    def __init__(self, config, func_kernel_size=0, global_align=False):
        super().__init__()
        self.fmri_len = 257
        self.len = 77
        self.enc_emb = 1024
        self.emb_size = 1024
        self.window_size = config['window_size']
        self.fmri2clip_cfg = config['fmri2clip_cfg']

        self.func_kernel_size = func_kernel_size
        if self.func_kernel_size > 1:
            self.func_align_layer = LocallyConnected2d(1,1,(256,256),func_kernel_size,1,global_align=global_align,patch_size=config['patch_size'])

        self.fmri_encoder      = Neural_fMRI2fMRI(config)
        self.fmri_encoder_proj = fmriEncoderProjection(len_in=self.fmri_len, len_out=self.len, enc_emb=self.enc_emb, emb_size=self.emb_size, window_size=self.window_size, dropout=config['drop_rate'], with_attn=config['with_attn'])

        self.logit_scale       = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        if self.fmri2clip_cfg == 'linear':
            self.fmri2clip     = nn.Sequential(
                nn.Dropout(config['drop_rate']),
                nn.Linear(self.window_size * self.enc_emb, self.emb_size, bias=True)
            )
        elif self.fmri2clip_cfg == 'fmri2clip':
            self.fmri2clip     = fmri2clip(dropout=config['drop_rate'])

        self.load_weight(config)

    def del_decoder(self):
        del self.fmri_encoder.transformer.pred
        del self.fmri_encoder.transformer.decoder_embed
        del self.fmri_encoder.transformer.mask_token
        del self.fmri_encoder.transformer.decoder_pos_embed
        del self.fmri_encoder.transformer.decoder_blocks
        del self.fmri_encoder.transformer.decoder_norm
        del self.fmri_encoder.transformer.decoder_pred

    def load_weight(self, config):
        if config['stage'] == 'fmri2text':
            if os.path.exists(config['pretrained_weight']):
                print('Loading %s Pre-trained weights ...' % (config['pretrained_weight']))
                pre_state_dict = torch.load(config['pretrained_weight'], map_location=torch.device('cpu'))
                model_state_dict = {k.replace('module.', ''): v for k, v in pre_state_dict['model'].items()}
                torch_init_model(self.fmri_encoder, model_state_dict)
            else:
                raise FileNotFoundError
            self.del_decoder()

        elif config['stage'] == 'fmri2video':
            self.del_decoder()
            if os.path.exists(config['pretrained_weight']):
                print('Loading %s Pre-trained weights ...' % (config['pretrained_weight']))
                pre_state_dict = torch.load(config['pretrained_weight'], map_location=torch.device('cpu'))
                model_state_dict = {k.replace('module.', ''): v for k, v in pre_state_dict['model'].items()}
                torch_init_model(self, model_state_dict)
            else:
                raise FileNotFoundError
            del self.fmri2clip
            del self.logit_scale
        else:
            raise NotImplementedError
        
    def forward(self, fmri):
        BN = fmri.shape[0]
        fmri = fmri.flatten(0, 1)

        if self.func_kernel_size > 1:
            fmri = self.func_align_layer(fmri)

        fmri_latent = self.fmri_encoder.transformer.forward_encoder_wo_pred(fmri) # batch_size*window_size(2) fmri_len enc_emb
        
        fmri_latent = fmri_latent.reshape(BN, self.window_size, self.fmri_len, self.enc_emb)

        fmri_embedding = self.fmri_encoder_proj(fmri_latent) # batch_size 77 1024

        outputs = {'fmri_embedding' : fmri_embedding}
        if hasattr(self, 'func_align_layer'):
            outputs['func_align_fmri'] = fmri
        if hasattr(self, 'fmri2clip'):
            if self.fmri2clip_cfg == 'linear':
                fmri_latent = self.fmri2clip(fmri_latent[:,:,0].reshape(BN, -1))
            elif self.fmri2clip_cfg == 'fmri2clip':
                fmri_latent = self.fmri2clip(fmri_embedding)
            fmri_latent = fmri_latent / fmri_latent.norm(p=2, dim=-1, keepdim=True)
            outputs['fmri_latent'] = fmri_latent
        return outputs

class fmriEncoderProjection(torch.nn.Module):
    def __init__(self, len_in=257, len_out=77, enc_emb=1024, emb_size=1024, window_size=2, num_heads=16, dropout=0, with_attn=False):
        super().__init__()

        self.len_out = len_out
        self.emb_size = emb_size
        self.window_size = window_size
        self.cond_state_transform = fmri2text(len_in=len_in*window_size, len_out=len_out, emb_size=emb_size, dropout=dropout)
        self.with_attn = with_attn

        if self.with_attn:
            self.attn = BlockTemp(dim=emb_size, num_heads=num_heads, mlp_ratio=1., qkv_bias=True, norm_layer=nn.LayerNorm, drop_path=dropout)

    def forward(self, x):
        # x : BN * window_size * len_in * enc_emb
        assert len(x.shape) == 4 and x.shape[1] == self.window_size
        if self.with_attn:
            x = rearrange(x, "b f d c -> (b f) d c")
            x = self.attn(x, window_size=self.window_size)
            x = rearrange(x, "(b f) d c -> b f d c", f=self.window_size)
        # x : BN * window_size * len_in * enc_emb
        x = rearrange(x, "b f d c -> b (f d) c")
        x = self.cond_state_transform(x)
        # x : BN * len_out * emb_size
        return x

class fmri2clip(nn.Module):
    def __init__(self, dropout=0, len_in=77, len_out=1, enc_dim=1024, cond_dim=1024):
        super().__init__()
        # prepare pretrained fmri mae 
        self.fmri_seq_len = len_in
        self.fmri_latent_dim = enc_dim
        self.channel_mapper = nn.Sequential(
            nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 4, 1, bias=False),
            nn.Dropout(dropout),
            nn.Conv1d(self.fmri_seq_len // 4, self.fmri_seq_len // 16, 1, bias=False),
            nn.Dropout(dropout),
            nn.Conv1d(self.fmri_seq_len // 16, len_out, 1, bias=False)
        )
        self.dim_mapper = nn.Linear(enc_dim, cond_dim, bias=True)

    def forward(self, x):
        # n, c, w = x.shape
        x = self.channel_mapper(x)
        x = self.dim_mapper(x)
        x = x.view(x.shape[0], -1)
        return x

class fmri2text(nn.Module):
    def __init__(self, 
                 len_in=257, 
                 len_out=77, 
                 enc_emb=1024, 
                 emb_size=1024, 
                 dropout=.0, 
                 use_norm=False,
                 use_norm_in=False,
                 ):
        super().__init__()
        self.projection = nn.Linear(enc_emb, emb_size, bias=False)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(emb_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_size) if use_norm else nn.Identity() 
        self.layer_norm_in = nn.LayerNorm(enc_emb) if use_norm_in else nn.Identity() 
        self.channel_mapper = nn.Sequential(
            nn.Linear(len_in, len_in//4, bias=False),
            nn.Linear(len_in//4, len_out, bias=False),
        )
        self.len_in = len_in
        self.len_out = len_out
    def forward(self, x):  
        x = self.layer_norm_in(x)
        x = x.transpose(1, 2)
        x = self.channel_mapper(x) 
        x = x.transpose(1, 2)
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x # n, out_channel, out_dim

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def save_model(model, opt, path, epoch, iter, prefix=None):
    if prefix is not None:
        save_path = path + "_{}.pth".format(prefix)
    else:
        save_path = path + ".pth"

    model_state = model_state_to_cpu(model.state_dict())
    print('\nsaving {}...\n'.format(save_path))
    all_saving = {'model': model_state,
                    # 'opt': opt.state_dict(),
                    'cur_epoch': epoch,
                    'accum_iter': iter}
    torch.save(all_saving, save_path)

def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # reshape to ncfhw
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    # unnormalize back to [0,1]
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # f h w c
    return images

class fMRIToVideoSDPipeline(TextToVideoSDPipeline):
    @torch.no_grad()
    def __call__(
        self,
        fmri_embedding: torch.FloatTensor,
        negative_embedding: torch.FloatTensor = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_images_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        batch_size = fmri_embedding.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        if negative_embedding is None:
            do_classifier_free_guidance = False
        else:
            do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        fmri_embedding = fmri_embedding.to(device=device)
        if negative_embedding is not None:
            fmri_embedding = torch.cat([negative_embedding, fmri_embedding])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            fmri_embedding.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=fmri_embedding,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            return TextToVideoSDPipelineOutput(frames=latents)

        video_tensor = self.decode_latents(latents)

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (video,)

        return TextToVideoSDPipelineOutput(frames=video)

class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.SafeLoader)
            self._dict['path'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


class Attention(nn.Module):
    _use_memory_efficient_attention_xformers = False
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, 
                                                     attention_op: Optional[Callable] = None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
        self._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self.attention_op = attention_op
        # print('fmri_encoder: use xformers attention')

    def batch_to_head_dim(self, tensor):
        head_size = self.num_heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    def head_to_batch_dim(self, tensor):
        head_size = self.num_heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor
     
    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # q: B, num_heads, N, C // num_heads
        # k: B, num_heads, N, C // num_heads
        # v: B, num_heads, N, C // num_heads
        if return_attn:
            assert not self._use_memory_efficient_attention_xformers, 'return_attn is not supported with xformers'
            assert not self.training, 'return_attn is not supported in training mode'
        if self._use_memory_efficient_attention_xformers:
            q = q.reshape(B * self.num_heads, N, C // self.num_heads)
            k = k.reshape(B * self.num_heads, N, C // self.num_heads)
            v = v.reshape(B * self.num_heads, N, C // self.num_heads)
            x = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op, scale=self.scale, p=self.drop_rate
            )
            x = x.to(q.dtype)
            x = self.batch_to_head_dim(x) # B, N, C
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if return_attn:
                return attn
            
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockTemp(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.attn_temp = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm_temp = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gradient_checkpointing = False
    
    def forward(self, x, window_size=None, return_attn=False, **kwargs):
        if return_attn:
            assert not self.training, 'return_attn is not supported in training mode'

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.attn), self.norm1(x)))
            # Temporal-Attention
            d = x.shape[1]
            x = rearrange(x, "(b f) d c -> (b d) f c", f=window_size)
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.attn_temp), self.norm_temp(x)))
            x = rearrange(x, "(b d) f c -> (b f) d c", d=d)
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.mlp), self.norm2(x)))
        else:    
            if return_attn:
                spatial_attn = self.attn(self.norm1(x), return_attn=return_attn)
                x = x + self.drop_path(self.attn(self.norm1(x)))
                d = x.shape[1]
                x = rearrange(x, "(b f) d c -> (b d) f c", f=window_size)
                temporal_attn = self.attn_temp(self.norm_temp(x), return_attn=return_attn)
                return spatial_attn, temporal_attn
            
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # Temporal-Attention
            d = x.shape[1]
            x = rearrange(x, "(b f) d c -> (b d) f c", f=window_size)
            x = x + self.drop_path(self.attn_temp(self.norm_temp(x)))
            x = rearrange(x, "(b d) f c -> (b f) d c", d=d)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CLIPModel_txt_img(torch.nn.Module):
    
    def __init__(self):
        super(CLIPModel_txt_img, self).__init__()

        # self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')

    @torch.no_grad()
    def forward(self, txt, img):
        # txt : list[str]
        # img : tensor(C*H*W)
        # img values : [0, 1]
        device = img.device

        img = torch.cat([self.preprocess(transforms.ToPILImage()(im)).unsqueeze(0) for im in img], 0).to(device, dtype=torch.float16)
        txt = self.tokenizer(txt).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(img)
            text_features = self.model.encode_text(txt)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features, image_features
