import os
import pandas as pd

from torchvision import transforms as T



from data.datasets import VideoFrameDataset
from configs import diffusion_2d_config
from models.diffusion_classifier.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Trainer, GaussianDiffusion, Unet


def main(config: diffusion_2d_config.Config):
    
    model = Unet(
        **config.unet_param
    )

    diffusion = GaussianDiffusion(
        model=model,
        image_size=config.image_size,
        **config.diffusion_param
    )


    transform = T.Compose([
        T.Resize((config.image_size,config.image_size)),
        T.RandomHorizontalFlip(config.aug_param['random_horizontal_flip_prob']),
        T.RandomVerticalFlip(config.aug_param['random_vertical_flip_prob']),
        T.RandomRotation(degrees=config.aug_param['random_rotation_degree']),
        T.RandomApply([
            T.GaussianBlur(kernel_size=config.aug_param['random_gaussian_blur_kernel'], sigma=config.aug_param['random_gaussian_blur_sigma'])
            ], p=config.aug_param['random_gaussian_blur_prob']),
        T.RandomAffine(degrees=config.aug_param['random_affine_degree']),
        T.ToTensor()
    ])

    
    PID = os.listdir(config.VIDEO_FRAME_DIR)
    TEST_ID = [f"Patient {id}" for id in config.specify_fold_test_patient[config.train_fold]]
    TRAIN_ID = [id for id in PID if id not in TEST_ID]

    print("=> Preparing Data with Patients:" , ' '.join(TRAIN_ID))
    video_path = []
    for ID in TRAIN_ID:
        images = os.listdir(os.path.join(config.VIDEO_FRAME_DIR, ID))
        images = [os.path.join(config.VIDEO_FRAME_DIR, ID, image) for image in images]
        video_path.extend(images)
    


    dataset = VideoFrameDataset(files=video_path, transforms=transform)


    results_folder = f'./diffusion_ckpt_{config.train_fold}'

    trainer = Trainer(
        diffusion_model = diffusion,
        dataset = dataset,
        train_batch_size = config.batch_size,
        results_folder = results_folder,
        **config.trainer_param
    )

    trainer.load(config.milestone)

    trainer.train()



if __name__ == "__main__":
    config_1 = diffusion_2d_config.Config()
    config_1.milestone = None
    config_1.train_fold = 'fold-1'
    main(config_1)

    config_2 = diffusion_2d_config.Config()
    config_2.milestone = None
    config_2.train_fold = 'fold-2'

    main(config_2)

    config_3 = diffusion_2d_config.Config()
    config_3.milestone = None
    config_3.train_fold = 'fold-3'

    main(config_3)

    config_4 = diffusion_2d_config.Config()
    config_4.milestone = None
    config_4.train_fold = 'fold-4'
    main(config_4)

    