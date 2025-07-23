import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.diffusion_classifier.resnet_3d import BasicBlock as Block_3d, Block_2d

from monai.networks.nets.vit import ViT
from models.diffusion_classifier.attention_head import Block as AttentionBlock

UNET_BLOCK_CHANNEL = [64,64,64,64,128,128,256,256,512,512,512,512,256,256,128,128,64,64]
UNET_SPATIENT_DIM  = [1,1,2,2,4,4,8,8,8,8,8,8,4,4,2,2,1,1]

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    


class AttentionHead(nn.Module):
    def __init__(self,
                 dim : int,
                 num_heads : int,
                 mlp_ratio : int,
                 num_blocks : int):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            layers.append(AttentionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            ))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)



class AttentionFusion(nn.Module):
    def __init__(self,
                 diffusion,
                 timestep :int,
                 blocks_to_use : list,
                 pool_size : int,
                 attention_dim : int,
                 dropout : float):
        
        super().__init__()
        
        self.diffusion = diffusion
        self.timestep = timestep
        self.blocks_to_use = blocks_to_use
        self.pool_size = pool_size

        self.freeze_diffusion_parameters()

        self.pre_layer = nn.ModuleList()

        for block_id in self.blocks_to_use:
            self.pre_layer.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.BatchNorm2d(UNET_BLOCK_CHANNEL[block_id]),
                    nn.Conv2d(UNET_BLOCK_CHANNEL[block_id], attention_dim, kernel_size=1),
                    LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c')),
                    nn.Dropout(dropout)
                )
            )

        self.attention_fusion_layer = AttentionHead(
            dim = attention_dim,
            num_heads = 4,
            num_blocks = 2,
            mlp_ratio = 2
        )
        

            
    def freeze_diffusion_parameters(self):
        for param in self.diffusion.parameters():
            param.requires_grad = False

    
    def forward(self, x_frame):
        blocks = self.diffusion.get_unet_features(x_frame, t=self.timestep, get_all = True)
        block_features = []

        for i , block_id in enumerate(self.blocks_to_use):
            x = self.pre_layer[i](blocks[block_id])
            block_features.append(x)

        x = torch.cat(block_features, dim=1)
        x = self.attention_fusion_layer(x)

        x = x.unsqueeze(1)

        return x





class Resnet3DHead(nn.Module):
    def __init__(self, 
                in_channels : int,
                planes_3d_blocks : tuple | list,
                pool_size : int,
                num_blocks : int,
                hidden_dim : int,
                n_classes : int,
                out_prediction : bool = True):
        
        super().__init__()
        self.in_channels = in_channels
        self.output_3d_layer = None
        self.hidden_dim = hidden_dim
        resnet_3d_layer = []

        for plane in planes_3d_blocks:
            resnet_3d_layer.append(
                self._make_layer(
                    ResBlock = Block_3d,
                    num_blocks = num_blocks,
                    plane = plane,
                    stride = 1,
                    is_2d=False,
            ))
            prev_channel = plane
            
        resnet_block_3d = nn.Sequential(*resnet_3d_layer)
        avgpool = nn.AdaptiveAvgPool3d((pool_size, pool_size, pool_size))
        last = nn.Linear(prev_channel * (pool_size ** 3), hidden_dim)

        self.output_3d_layer = nn.Sequential(
            resnet_block_3d,
            avgpool,
            nn.Flatten(start_dim=1),
            last,
        )

    def forward(self, x):
        return self.output_3d_layer(x)
    
    def _make_layer(self, ResBlock, num_blocks, plane, stride=1 , is_2d=True):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != plane*ResBlock.expansion:
            if is_2d: # 2d
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, plane*ResBlock.expansion, kernel_size=1, stride = stride),
                    nn.BatchNorm2d(plane*ResBlock.expansion)
                )
            else: # 3d
                downsample = nn.Sequential(
                    nn.Conv3d(self.in_channels, plane * ResBlock.expansion,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm3d(plane * ResBlock.expansion)
                )
        
        layers.append(ResBlock(self.in_channels, plane, downsample = downsample, stride = stride))
        self.in_channels = plane * ResBlock.expansion

        for i in range(num_blocks - 1):
            layers.append(ResBlock(self.in_channels, plane))

        return nn.Sequential(*layers)
    




class DiffusionFeatureBaseModule(nn.Module):
    def __init__(self,  diffusion ):
        
        super().__init__()

        self.diffusion = diffusion

        self.freeze_diffusion_parameters()
    
    def freeze_diffusion_parameters(self):
        print("=> Freeze Diffusion Model Parameters!")
        for param in self.diffusion.parameters():
            param.requires_grad = False


class ResnetFusion(DiffusionFeatureBaseModule):
    def __init__(self,
                 image_size, 
                 diffusion,
                 hidden_channels : int,
                 planes_2d_blocks : tuple | list,
                 timestep : int,
                 blocks_to_use : list, # range from 0-17
                 num_blocks : int,
                 get_all_features : bool = False,
                 max_pool : bool = True,
                 use_resnet_2d : bool = True,
                 fusion_method : str = "handamard"):
        
        super().__init__(diffusion)

        self.image_size = image_size
        self.timestep = timestep
        self.blocks_to_use = blocks_to_use
        self.get_all_features = get_all_features # total 19 blocks of features
        self.max_pool = max_pool
        self.hidden_channels = hidden_channels

        # the channel of each block from the diffusion model
        
        assert all(0 <= block_num < len(UNET_BLOCK_CHANNEL) for block_num in self.blocks_to_use), f"Blocks_to_use must be in range 0 ~ {len(UNET_BLOCK_CHANNEL)-1}"

        assert fusion_method in ["handamard" , "addition"], "Fusion Method not implemented"
        self.fusion_method = fusion_method


        self.in_channels = self.hidden_channels
        self.block_layers_2d = nn.ModuleList()

        if use_resnet_2d: # use 2d resnet to change the channel dimension
            for block in blocks_to_use:
                self.block_layers_2d.append(self._make_2d_resnet_layer(
                    ResBlock = Block_2d,
                    unet_block = block,
                    num_blocks = num_blocks,
                    planes_2d_blocks = planes_2d_blocks,
                    stride = 1,
                ))
        else: # 1x1 conv to change the channel dimension
            for block in blocks_to_use:
                self.block_layers_2d.append(self._make_2d_conv(block))


      


    def _make_2d_conv(self, unet_block):
        first_conv = nn.Sequential(
            # Downsample
            nn.Conv2d(in_channels=UNET_BLOCK_CHANNEL[unet_block],
                      out_channels=self.hidden_channels,
                      kernel_size=7, 
                      stride=2, 
                      padding=3, 
                      bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(),
            # Downsample twice
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1) if self.max_pool else nn.Identity()
        )

        return first_conv

    def _make_2d_resnet_layer(self, ResBlock, unet_block, num_blocks, planes_2d_blocks, stride = 1):
        # Do we need to downsample twice with max pooling? Max Pooling reduce GPU memory usage
        first_conv = nn.Sequential(
            # Downsample
            nn.Conv2d(in_channels=UNET_BLOCK_CHANNEL[unet_block],
                      out_channels=self.hidden_channels,
                      kernel_size=7, 
                      stride=2, 
                      padding=3, 
                      bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(),
            # Downsample twice
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1) if self.max_pool else nn.Identity()
        )

        resnet_2d_layer = []

        self.in_channels = self.hidden_channels

        for plane in planes_2d_blocks:
            resnet_2d_layer.append(
                self._make_layer(
                    ResBlock = ResBlock,
                    num_blocks = num_blocks,
                    plane = plane,
                    stride = stride,
                    is_2d=True
            ))

        resnet_2d_layer = nn.Sequential(*resnet_2d_layer)

        first_conv.append(resnet_2d_layer)
        return first_conv


    
    def _make_layer(self, ResBlock, num_blocks, plane, stride=1 , is_2d=True):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != plane*ResBlock.expansion:
            if is_2d: # 2d
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, plane*ResBlock.expansion, kernel_size=1, stride = stride),
                    nn.BatchNorm2d(plane*ResBlock.expansion)
                )
            else: # 3d
                downsample = nn.Sequential(
                    nn.Conv3d(self.in_channels, plane * ResBlock.expansion,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm3d(plane * ResBlock.expansion)
                )
        
        layers.append(ResBlock(self.in_channels, plane, downsample = downsample, stride = stride))
        self.in_channels = plane * ResBlock.expansion

        for i in range(num_blocks - 1):
            layers.append(ResBlock(self.in_channels, plane))

        return nn.Sequential(*layers)


    def forward(self, x_frame):
        """Is there a way to make it faster and more efficient?"""

        blocks = self.diffusion.get_unet_features(x_frame, t=self.timestep, get_all = self.get_all_features)

        # combine encoder feature and decoder feature after resnet 2d blocks
        out_features = []
        for i, block_id in enumerate(self.blocks_to_use):
            out_features.append(self.block_layers_2d[i](blocks[block_id]))

        out = self.fusion(out_features)

        return out



    def fusion(self, x):
        out = None
        if self.fusion_method == 'addition':
        # 2. additional feature fusion
            out = torch.zeros_like(x[0])
            for out_feature in x:
                out = out + out_feature

        # 3. Handamard Product feature fusion
        elif self.fusion_method == 'handamard':
            out = torch.ones_like(x[0])
            for out_feature in x:
                out = out * out_feature

        return out


class DiffusionEncoder(DiffusionFeatureBaseModule):
    def __init__(self,
                image_size, 
                diffusion,
                hidden_chn : int,
                timestep : int,
                block_to_use : int,  # range from 0-17
                max_pool : bool = False): 
        
        
        super().__init__(diffusion)

        self.image_size = image_size
        self.timestep = timestep
        self.block_to_use = block_to_use
        

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=UNET_BLOCK_CHANNEL[block_to_use],
                        out_channels=hidden_chn,
                        kernel_size=7, 
                        stride=2, 
                        padding=3, 
                        bias=False),
            nn.BatchNorm2d(hidden_chn),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1) if max_pool else nn.Identity()
        )

    def forward(self, x_frame):
        """Using only single block output feature as the feature representation"""
        blocks = self.diffusion.get_unet_features(x_frame, t=self.timestep, get_all = True)

        out = self.conv(blocks[self.block_to_use])
        return out




