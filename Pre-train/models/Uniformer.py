import torch
import torch.nn as nn
from models.uniformer_blocks import uniformer_small
from monai.utils import ensure_tuple_rep
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from typing import List
from torch import Tensor
from utils.ops import aug_rand_with_learnable_rep
import torch.nn.functional as F
from losses.loss import Contrast_Loss
from models.attention_pooling import GatedAttention
from collections import OrderedDict

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = patch_size
        else:
            stride = stride
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # print ('conv3d: ', x.shape)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class SSLEncoder(nn.Module):
    def __init__(self, num_phase: int, initial_checkpoint: str=None):
        super().__init__()
        self.uniformer = uniformer_small(in_chans=num_phase)
    def forward(self, x):
        x_0, x_enc1, x_enc2, x_enc3, x_enc4 = self.uniformer(x)
        return x_0, x_enc1, x_enc2, x_enc3, x_enc4
 

class UniSegDecoder(nn.Module):
    def __init__(self, img_size: int, in_chans: int,cls_chans=0):
        super().__init__()
        self.decoder5 = UnetrUpBlock(
                    spatial_dims=3,
                    in_channels=512,
                    out_channels=320,
                    kernel_size=3,
                    # upsample_kernel_size=2,
                    upsample_kernel_size=(1,2,2),
                    norm_name="instance",
                    res_block=True,
                )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=320,
            out_channels=128,
            kernel_size=3,
            # upsample_kernel_size=2,
            upsample_kernel_size=(1,2,2),
            norm_name="instance",
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            # upsample_kernel_size=2,
            upsample_kernel_size=(1,2,2),
            norm_name="instance",
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            # upsample_kernel_size=2,
            upsample_kernel_size=(1,2,2),
            norm_name="instance",
            res_block=True,
        )

        # self.proj1 = PatchEmbed(
        #         img_size=img_size, patch_size=(1,3,3), in_chans=in_chans, embed_dim=64, stride=1, padding=(0,1,1))    
        
        # NOTE in ds this part is replaced with decoder2
        # self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        if cls_chans==0:
            self.out_1 = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=in_chans)
        else:
            self.out_1 = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=cls_chans)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x0, x1, x2, x3, x4):
        # we do not use skip connection for better representation learning
        
        dec5 = self.decoder5.transp_conv(x4.permute(0,1,3,4,2))
        dec4 = self.decoder4.transp_conv(dec5) 
        dec3 = self.decoder3.transp_conv(dec4)
        dec2 = self.decoder2.transp_conv(dec3)
        x_rec = self.out_1(dec2) # 4

        return x_rec


    
class RecModel(nn.Module):
    def __init__(self, args, dim=768):
        super(RecModel, self).__init__()
        self.device = args.device
        in_chans = args.in_channels
        img_size = args.roi_x
        self.encoder = SSLEncoder(num_phase=in_chans, initial_checkpoint=args.initial_checkpoint)
        # self.decoder = UniSegDecoder(img_size=img_size, in_chans=in_chans)
        # B N M to B M
        self.pool_layers = nn.Sequential(
        OrderedDict(
            [
                ("relu", nn.ReLU()),
                ("pool", nn.AdaptiveAvgPool3d(1)),
                ("flatten", nn.Flatten(1)),
                        ]
                    )
                )
        self.pool_non_cine = GatedAttention(in_dim=512,embed_dim=512)
        self.pool_img = GatedAttention(in_dim=512,embed_dim=512)
        self.projector = nn.Sequential(
                nn.Linear(in_features=512,out_features=512,bias=True),
                nn.ReLU(),
                nn.Linear(in_features=512,out_features=128,bias=True),
        )
        # num_modals = len(args.template_index)
        # we found use zero-init will accelerate template learning and produce better visiability
        # non-cine squence
        # self.rep_template = nn.Parameter(torch.zeros((6,args.dst_h, args.dst_w, args.dst_d)))
        
        self.kl_loss = Contrast_Loss(temperature=1.0)
        self.recon_loss = nn.MSELoss(reduction="mean")
        
    def forward(self, batch):
        # CINE
        # print(batch['cinesa'].shape)
        # print(batch['non_cine'].shape)
        # print(batch['mod_parent'])
        # print(batch['cinesa'].shape,batch['non_cine'].shape)
        cine_rpt = self.encoder(batch['cinesa'])[-1] # [3, 512, 25, 6, 6]
        B,dim,D,H,W = cine_rpt.shape
        # cine_rpt = cine_rpt.contiguous().view(B, D * H * W, 512)  
        cine_rpt = self.pool_layers(cine_rpt) # [B, 512]
        # print(cine_rpt.shape) 
        # non cine
        B,mod_num,c,h,w,d = batch['non_cine'].shape
        non_cine_inputs = batch['non_cine'].contiguous().view(B*mod_num,c,h,w,d)
        
        non_cine_rpt = self.encoder(non_cine_inputs)[-1] # [B*2, 512, 3, 6, 6]
        non_cine_rpt = self.pool_layers(non_cine_rpt) # [B*2, 512]
        
        non_cine_rpt  = non_cine_rpt.contiguous().view(B,mod_num,512)# 
        non_cine_rpt,score_non_cine = self.pool_non_cine(non_cine_rpt)
        # print(non_cine_rpt.shape)
        
        # representation aggregation
        img_rpt = torch.stack((cine_rpt,non_cine_rpt),dim=1)
        img_rpt,score_mod = self.pool_img(img_rpt)
        # print(img_rpt.shape)
        # we need to close its representation to each component
        # print(img_rpt.shape)
        loss_center = self.kl_loss(img_rpt,cine_rpt) + self.kl_loss(img_rpt,non_cine_rpt)
        
        # contrastive learning
        cine_embed = self.projector(cine_rpt)
        non_cine_embed = self.projector(non_cine_rpt)
        # print(cine_embed.shape,non_cine_embed.shape)
        loss_mod = self.kl_loss(cine_embed,non_cine_embed)
        
        return [loss_center,loss_mod],[score_non_cine,score_mod],img_rpt
    
if __name__=="__main__":
    data = torch.randn((2, 512, 13, 14, 14))
    decoder = UniSegDecoder(img_size=256, in_chans=256)
    out = decoder(data)
    