def main():
    from configs.classification_config import DiffusionClassifierConfig
    from configs.classification_config import Config
    from trainer import Trainer


    config = DiffusionClassifierConfig()
    # config = Config()
    config.result_dir = './results/vde_ppt_2'
    config.checkpoint_dir =  './checkpoints/vde_ppt_2'

    # config = Config()
    # config.model = "videoMAE"
    # config.result_dir = "./results/videoMAE"
    # config.checkpoint_dir = "./checkpoints/videoMAE"
    # config.channel_first = False


    trainer = Trainer(config)
    trainer.train_cross_validation()


if __name__ == '__main__':
    main()