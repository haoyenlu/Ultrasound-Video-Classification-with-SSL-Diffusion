class Config:
    image_size = 224

    VIDEO_DIR = 'toronto_ssim_kneedle'
    PATIENT_CSV = 'patient_info.csv'

    mode = 'train' # train or test
    model = 'vivit'
    load_pretrained = True
    
    window_step_size = 2
    batch_size = 8
    num_frames = 12
    num_classes = 1
    num_channels = 1
    sample_ratio = 0.8
    sample_method = 'window' # window or random_sample


    num_epoch = 20
    learning_rate = 1e-4

    result_dir = './results/resnet50_new'
    checkpoint_dir = './checkpoints/resnet50_new'


    folds_to_train = [1,2,3,4]
    repeat_cross_validate = 10

    continue_training = False

    specify_fold_test_patient = {
        "fold-1": ["008","006","011","012","013","021"],
        "fold-2": ["005","014","015","017","019","024"],
        "fold-3": ["003","007","016","018","022","023"],
        "fold-4": ["001","002","004","009","010","020"]
    }


    augmentation = True
    augmentation_prob = 0.7
    rotation_degree = 20
    gaussian_blur_sigma = 1
    translation_degree = 20

    channel_first = True # False when using hugging face model (Vivit, VideoMAE)



class DiffusionClassifierConfig(Config):
    model = 'Diffusion'

    feature_fusion_method = 'Encoder' # Resnet or Attention or Encoder


    """Diffusion Model Parameters"""
    timestep = 50

    diffusion_fold_checkpoint = {
        'fold-1': 'diffusion_ckpt_fold-1/model-20.pt',
        'fold-2': 'diffusion_ckpt_fold-2/model-20.pt',
        'fold-3': 'diffusion_ckpt_fold-3/model-20.pt',
        'fold-4': 'diffusion_ckpt_fold-4/model-20.pt'
    }
    
    # Backbone Model
    unet_param = {
        'dim': 64,
        'dim_mults': (1,2,4,8),
        'channels': 1,
        'attn_heads': 4
    }
    # diffusion
    diffusion_param = {
        'timesteps': 1000,
        'beta_schedule': 'cosine'
    }



    # Classification Model
    attn_param = {    
        'attn_pool_size': 4,
        'attention_dim': 64,
        'dropout': 0.6,
        'blocks_to_use': [10, 11 ,12, 13]
    }

    resnet_param = {
        'blocks_to_use': [10, 11 ,12, 13],
        'hidden_channels': 64,
        'planes_2d_blocks': [64,128,256],
        'num_blocks': 2,
        'max_pool': True,
        'fusion_method': 'addition',
    }

    encoder_param = {
        'block_to_use': 13,
        'max_pool': True,
        'hidden_chn': 16
    }


    resnet_head_param = {
        'in_channels': 16,
        'planes_3d_blocks': [32, 64, 128],
        'pool_size': 2,
        'hidden_dim': 128,
        'num_blocks' : 2,
    }

    demo_param = {
        'sex_embedding': 4,
        'age_embedding': 4,
        'ppt_embedding': 4,
        'num_ppt': 2,
        'n_classes': 1
    }


