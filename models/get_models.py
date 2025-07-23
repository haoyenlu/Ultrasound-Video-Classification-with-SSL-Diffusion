import torch.nn as nn
import torch


from transformers import VivitConfig, VivitForVideoClassification
from transformers import VideoMAEForVideoClassification, VideoMAEConfig


from configs.classification_config import Config, DiffusionClassifierConfig

def get_default_vivit(config: Config):
    

    model_ckpt = "google/vivit-b-16x2-kinetics400"

    model_config = VivitConfig(
        image_size=config.image_size,
        num_frames=config.num_frames,
        patch_size=16,
        num_channels=config.num_channels,
        num_attention_heads=4,
        num_hidden_layers=8,
        # intermediate_size=256,
        # hidden_dropout_prob=0.5
    )

    if config.load_pretrained:
        model = VivitForVideoClassification(model_config).from_pretrained(
            model_ckpt,
            ignore_mismatched_sizes=True,
            config=model_config
        )
    else:
        model = VivitForVideoClassification(model_config)

    model.classifier = nn.Linear(model.classifier.in_features, config.num_classes)

    return model



def get_default_resnet50(config: Config):    
    from models.generate_model import generate_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = generate_model(model_depth=50,
                           model_name='resnet',
                           n_classes=config.num_classes,
                           sample_size=config.image_size,
                           sample_duration=config.num_frames)
    
    checkpoint_path = './resnet-checkpoint/resnet-50-kinetics.pth'
    model_data = torch.load(checkpoint_path, map_location=device)
    state_dict = model_data['state_dict']
    state_dict_new = {k.replace("module.",""):v for k,v in state_dict.items()}
    for k in list(state_dict_new.keys()):
        if k.startswith("fc."):
            del state_dict_new[k]


    model.load_state_dict(state_dict_new,strict=False)

    return model



def get_default_resnet101(config: Config):
    
    from models.generate_model import generate_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = generate_model(model_depth=101,
                        model_name='resnet',
                        n_classes=config.num_classes,
                        sample_size=config.image_size,
                        sample_duration=config.num_frames)

    checkpoint_path = './resnet-checkpoint/resnet-101-kinetics.pth'
    model_data = torch.load(checkpoint_path, map_location=device)
    state_dict = model_data['state_dict']
    state_dict_new = {k.replace("module.",""):v for k,v in state_dict.items()}
    for k in list(state_dict_new.keys()):
        if k.startswith("fc."):
            del state_dict_new[k]

    model.load_state_dict(state_dict_new,strict=False)


    return model




def get_default_videoMAE(config: Config):
    

    model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
    model_config = VideoMAEConfig(
        image_size=config.image_size,
        num_frames=config.num_frames,
        patch_size=8,
        num_channels=config.num_channels,
        num_attention_heads=4,
        num_hidden_layers=4,
        # intermediate_size=256,
        # hidden_dropout_prob=0.5
    )

    model = VideoMAEForVideoClassification(model_config).from_pretrained(
        model_ckpt,
        ignore_mismatched_sizes=True,
        config=model_config
    )


    model.classifier = nn.Linear(model.classifier.in_features, config.num_classes)

    return model





def get_default_diffusion_classifier(fold, config: DiffusionClassifierConfig):
    from models.diffusion_classifier.get_model import get_diffusion_video_classifier

    model = get_diffusion_video_classifier(fold, config)

    return model

