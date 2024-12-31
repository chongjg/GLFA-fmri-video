import torch
import torch.nn as nn

import numpy as np
from timm.models.vision_transformer import Block as TransBlock

def get_2d_sincos_pos_embed(embed_dim, grid_size=(), cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class PatchEmbed2D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=1, embed_dim=768, flatten=True, bias=True):
        super().__init__()
        assert isinstance(img_size, tuple)
        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten
        assert img_size[0] % patch_size == 0
        assert img_size[1] % patch_size == 0

        self.num_h = img_size[0] // patch_size
        self.num_w = img_size[1] // patch_size
        self.num_patches = self.num_h * self.num_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape # batch, channel, voxels
        assert H == self.img_size[0]
        assert W == self.img_size[1]

        x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous() # NCHW -> NLC

        return x # [B, L, C]

class fMRI_Autoencoder(nn.Module):

    def __init__(self, config):
        super(fMRI_Autoencoder, self).__init__()
        patch_size = config['patch_size']
        image_size = tuple(config['image_size'])

        # num_voxel = config.Data['num_voxel']
        embed_dim = config['embed_dim']
        decoder_embed_dim = config['decoder_embed_dim']
        num_head = config['num_heads']
        drop_p = config['drop_rate']
        in_chans = config['in_chans']
        img_dim = 2048

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.PatchEmbed2DArgs = [image_size, patch_size, in_chans, embed_dim]
        self.patch_embed = PatchEmbed2D(image_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            TransBlock(embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True,
                       drop=drop_p, attn_drop=drop_p, drop_path=drop_p, norm_layer=nn.LayerNorm)
            for _ in range(config['encoder_depth'])])
        self.norm = nn.LayerNorm(embed_dim)

        self.pred = nn.Linear(embed_dim, img_dim, bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(img_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            TransBlock(decoder_embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True,
                       drop=drop_p, attn_drop=drop_p, drop_path=drop_p, norm_layer=nn.LayerNorm)
            for _ in range(config['decoder_depth'])])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        # --------------------------------------------------------------------------
        self.initialize_weights()
        self.in_chans = in_chans

    def initialize_weights(self, ):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.patch_embed.num_h, self.patch_embed.num_w), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.patch_embed.num_h, self.patch_embed.num_w), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size
        h = self.patch_embed.num_h
        w = self.patch_embed.num_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))
        return imgs

    def forward_encoder_wo_pred(self, x):
        # x: [B, C, H ,W]
        patch_embed_out = self.patch_embed(x) # [B, K, C]

        # add pos embed w/o cls token
        x = patch_embed_out + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x#, patch_embed_out

    def forward_encoder_w_pred(self, x):
        # x: [B, C, H ,W]
        x = self.patch_embed(x) # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        z = self.pred(x)
        return z

    def forward_encoder(self, x, batch_cnt=None):
        # x: [B, C, H ,W]
        x = self.patch_embed(x) # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        z = self.pred(x[:, :1])
        return z
        # return x[:, :1]

    def forward_decoder(self, x, batch_cnt=None):
        # x: [B, 1, C]
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], self.patch_embed.num_patches, 1)
        z = torch.cat([x, mask_tokens], dim=1)  # append cls token

        # add pos embed
        z = z + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            z = blk(z) # [B, 1+K, C]
        z = self.decoder_norm(z) # [B, 1+K, C]

        # predictor projection
        out = self.decoder_pred(z) # [B, 1+K, p*p*3]
        return out[:, 1:, :] # [B, K, p*p*3]

    def forward(self, fmri, mask=None, target=None, batch_cnt=None):
        # fmri: [B, 1, K]

        latent = self.forward_encoder(fmri, batch_cnt)
        rec = self.forward_decoder(latent, batch_cnt)
        if target is not None:
            fmri = target
        loss = self.calculate_loss(fmri, rec, mask=mask)

        return rec, loss

    def calculate_loss(self, target, pred, mask=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, H, W], 0 is keep, 1 is remove,
        """
        pred = self.unpatchify(pred) # [N, 1, H, W]

        loss = (pred - target) ** 2 # [N, 1, H, W]
        loss = loss.flatten(start_dim=1)

        if mask is not None:
            mask = mask[:, None].repeat(1, pred.shape[1], 1, 1).flatten(start_dim=1) # [N, L]
            loss = (loss * mask).sum(-1) / mask.sum(-1)
            
        else:
            loss = loss.mean(-1)

        # return loss.mean()
        return loss

class Neural_fMRI2fMRI(nn.Module):
    def __init__(self, cfg):
        super(Neural_fMRI2fMRI, self).__init__()

        self.transformer = fMRI_Autoencoder(cfg)

    def fmri_projection_regloss(self, scale=0.05):
        return 0.5 * scale * (self.transformer.patch_embed.voxel_proj.weight ** 2).sum()

    def fmri_projection(self, fmri): # BN * 1 * 256 * 256
        assert len(fmri.shape) == 4
        fmri_patch = self.transformer.patchify(fmri) # BN * 256 * 256
        fmri_projection = fmri_patch + self.transformer.patch_embed.voxel_proj(fmri_patch)
        fmri_unpatch = self.transformer.unpatchify(fmri_projection)
        return fmri_unpatch

    def forward(self, image, target=None, mask=None, batch_cnt=None):
        # fmri: [B, 3, H, W]
        # transformer
        rec, loss = self.transformer(image, mask, target=target, batch_cnt=batch_cnt) # [B, L, p*p*3]

        return rec, loss 
   
    @torch.no_grad()
    def encode_feats(self, image):
        # fmri: [B, 3, H, W]
        B, _, H, W = image.size()

        # transformer
        feats = self.transformer.forward_encoder_wo_pred(image) # [B, L, C], before pred

        return feats