import torch
from models.diffusion_classifier.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion

from configs.classification_config import DiffusionClassifierConfig


def load_diffusion_model(fold, config: DiffusionClassifierConfig):
    print("=> Loading Diffusion Model Checkpoint from {}!".format(config.diffusion_fold_checkpoint[fold]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = Unet(
        **config.unet_param
    )

    diffusion = GaussianDiffusion(
        model=backbone,
        image_size=config.image_size,
        **config.diffusion_param
    )


    data = torch.load(config.diffusion_fold_checkpoint[fold], map_location=device, weights_only=True)
    diffusion.load_state_dict(data['model'])

    return diffusion



UNET_BLOCK_CHANNEL = [64,64,64,64,128,128,256,256,512,512,512,512,256,256,128,128,64,64]
UNET_SPATIENT_DIM  = [1,1,2,2,4,4,8,8,8,8,8,8,4,4,2,2,1,1]

def get_diffusion_video_classifier(fold, config : DiffusionClassifierConfig):
    from models.diffusion_classifier.classification_head import Resnet3DHead, ResnetFusion, AttentionFusion , DiffusionEncoder
    from models.diffusion_classifier.video_classifier import DiffusionVideoClassifier

    diffusion_model = load_diffusion_model(fold, config)

    diffusion_2d_feature_model = None

    if config.feature_fusion_method == 'Resnet':
        diffusion_2d_feature_model = ResnetFusion(
            image_size = config.image_size,
            diffusion=diffusion_model,
            timestep=config.timestep,
            get_all_features=True,
            **config.resnet_param
        )

    elif config.feature_fusion_method == 'Attention':
        diffusion_2d_feature_model = AttentionFusion(
            diffusion=diffusion_model,
            timestep=config.timestep,
            **config.attn_param            
        )

    elif config.feature_fusion_method == 'Encoder':
        diffusion_2d_feature_model = DiffusionEncoder(
            image_size = config.image_size,
            diffusion=diffusion_model,
            timestep=config.timestep,
            **config.encoder_param
        )

    else :
        raise ValueError(f"{config.feature_fusion_method} Fusion Method currently not supported.")
    


    classifier_head = Resnet3DHead(
        n_classes=config.num_classes,
        **config.resnet_head_param
    )

    

    model = DiffusionVideoClassifier(
        num_frames=config.num_frames,
        diffusion_2d_model=diffusion_2d_feature_model,
        classifier_head=classifier_head,
        **config.demo_param
    )

    return model

