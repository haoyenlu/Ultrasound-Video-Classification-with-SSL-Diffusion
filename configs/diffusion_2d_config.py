class Config:
    image_size = 224
    batch_size = 8


    VIDEO_FRAME_DIR = 'cropped_toronto_frame'
    PATIENT_CSV = 'patient_info.csv'

    milestone = None

    # patient
    train_fold = 'fold-1'
    specify_fold_test_patient = {
        "fold-1": ["008","006","011","012","013","021"],
        "fold-2": ["005","014","015","017","019","024"],
        "fold-3": ["003","007","016","018","022","023"],
        "fold-4": ["001","002","004","009","010","020"]
    }

    # unet
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

    # trainer 
    trainer_param = {
        'gradient_accumulate_every':4,
        'train_lr': 1e-4,
        'train_num_steps': 10000,
        'save_and_sample_every':500,
        'num_samples':4,
        'num_fid_samples': 4,
    }

    aug_param = {
        'random_horizontal_flip_prob': 0.7,
        'random_vertical_flip_prob': 0.7,
        'random_rotation_degree': 20,
        'random_gaussian_blur_prob': 0.5,
        'random_gaussian_blur_sigma': (0.1,2.0),
        'random_gaussian_blur_kernel': 3,
        'random_affine_degree': 20
    }