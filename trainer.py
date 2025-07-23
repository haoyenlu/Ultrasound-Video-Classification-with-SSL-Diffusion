import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from data.transforms import Compose, Scale, ToNumpy , ExpandDims
from models import get_models
from data.vidaug import augmentors as va

import os

from data.datasets import VideoDataset

from configs.classification_config import Config
from evaluate import Metric


def sex_transform(sex):
    if sex == 'M': return 0 
    elif sex == 'F': return 1
    else: raise ValueError(f"Unknown sex value: {sex}")


# ======================
# Classfication Trainer
# ======================

class Trainer:
    def __init__(self, config: Config):

        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Read Patient information
        self.patient_info = pd.read_csv(self.config.PATIENT_CSV)
        self.patient_info['label'] = np.where(self.patient_info['Type'] == 'Healthy', 0, 1)
        self.patient_ids = self.patient_info['Study ID'].tolist()
        
        self.specify_fold_test_patient = self.config.specify_fold_test_patient

        print(f"Training with {config.model} Model.")

    def __load__model(self, fold):
        if self.config.model == 'Resnet50':
            model = get_models.get_default_resnet50(self.config)

        elif self.config.model == 'Resnet101':
            model = get_models.get_default_resnet101(self.config)

        elif self.config.model == 'vivit':
            model = get_models.get_default_vivit(self.config)

        elif self.config.model == 'videoMAE':
            model = get_models.get_default_videoMAE(self.config)

        elif self.config.model == 'Diffusion':
            model = get_models.get_default_diffusion_classifier(fold, self.config)
        else:
            raise ValueError(f"Model {self.config.model} is not supported.")

        return model


    def __get__transform(self):
        spatial_transform = Compose([
            Scale(size=(self.config.image_size,self.config.image_size)),
            ToNumpy(),
            ExpandDims(),
        ])

        return spatial_transform

    def __get__augmentation(self):
        sometimes = lambda aug: va.Sometimes(self.config.augmentation_prob, aug)

        augmentation = va.Sequential([
            sometimes(va.HorizontalFlip()),
            sometimes(va.VerticalFlip()),
            sometimes(va.RandomRotate(self.config.rotation_degree)),
            sometimes(va.GaussianBlur(sigma=self.config.gaussian_blur_sigma)),
            sometimes(va.RandomTranslate(x=self.config.translation_degree,y=self.config.translation_degree))
        ])

        return augmentation

    def __get__dataloader(self , train_ids, test_ids):
        from sklearn.preprocessing import StandardScaler
        from collections import defaultdict

        video_data, video_label = defaultdict(list), defaultdict(list)
        sex_feature, age_feature = defaultdict(list), defaultdict(list)
        ppt_feature = defaultdict(list)

        # Include demographic (sex, age) and clinical (ppt) feature
        # PPT feature includes the average value of pre-intervention and post-intervention PPT 
        PIDS = {'train':train_ids, 'test':test_ids}
        for K in ['train','test']:
            for pid in PIDS[K]:
                label = self.patient_info.loc[self.patient_info['Study ID'] == pid,'label'].values[0]
                sex = sex_transform(self.patient_info.loc[self.patient_info['Study ID'] == pid, 'Sex'].values[0])
                age = self.patient_info.loc[self.patient_info['Study ID'] == pid, 'Age'].values[0]
                ppt_post_1 = self.patient_info.loc[self.patient_info['Study ID'] == pid, 'PPT_POST_1'].values[0]
                ppt_post_2 = self.patient_info.loc[self.patient_info['Study ID'] == pid, 'PPT_POST_2'].values[0]
                ppt_post_3 = self.patient_info.loc[self.patient_info['Study ID'] == pid, 'PPT_POST_3'].values[0]
                ppt_post_avg = self.patient_info.loc[self.patient_info['Study ID'] == pid, 'PPT_POST_AVG'].values[0]
                

                ppt_pre_1 = self.patient_info.loc[self.patient_info['Study ID'] == pid, 'PPT_PRE_1'].values[0]
                ppt_pre_2 = self.patient_info.loc[self.patient_info['Study ID'] == pid, 'PPT_PRE_2'].values[0]
                ppt_pre_3 = self.patient_info.loc[self.patient_info['Study ID'] == pid, 'PPT_PRE_3'].values[0]
                ppt_pre_avg = self.patient_info.loc[self.patient_info['Study ID'] == pid, 'PPT_PRE_AVG'].values[0]


                videos = [os.path.join(self.config.VIDEO_DIR, pid, v) for v in os.listdir(os.path.join(self.config.VIDEO_DIR, pid))]
                video_data[K].extend(videos)
                video_label[K].extend([label] * len(videos))
                sex_feature[K].extend([sex] * len(videos))
                age_feature[K].extend([age] * len(videos))
                ppt_feature[K].append([ppt_pre_avg, ppt_post_avg] * len(videos)) # using the average of both pre and post PPT 

        # Standardize Age Feature
        scaler = StandardScaler()
        age_feature['train'] = scaler.fit_transform(np.array(age_feature['train']).reshape(-1, 1)).flatten().tolist()
        age_feature['test']  = scaler.transform(np.array(age_feature['test']).reshape(-1, 1)).flatten().tolist()
        
        N, n = len(age_feature['train']), len(age_feature['test'])

        # Standarduze PPT Featyre
        ppt_scaler = StandardScaler()
        ppt_feature['train'] = ppt_scaler.fit_transform(np.array(ppt_feature['train'])).reshape(N,2)
        ppt_feature['test']  = ppt_scaler.transform(np.array(ppt_feature['test'])).reshape(n,2)


        datasets = {}
        for K in ['train','test']:
            datasets[K] = VideoDataset(video_data[K] , 
                                       video_label[K], 
                                        sex_feature=sex_feature[K],age_feature=age_feature[K],
                                        ppt_feature =ppt_feature[K],
                                        spatial_transform=self.__get__transform(),
                                        augmentation=self.__get__augmentation() if K == 'train' else None, 
                                        num_frames=self.config.num_frames, 
                                        step_size=self.config.window_step_size,
                                        channel_first=self.config.channel_first)
            

        train_dataloader = DataLoader(datasets['train'], self.config.batch_size, shuffle=True , num_workers=4)
        test_dataloader  = DataLoader(datasets['test'],  self.config.batch_size, shuffle=False , num_workers=4)
        return train_dataloader, test_dataloader


    """
    Repeated and Straitified Cross-validation training
    """
    def train_cross_validation(self):
        os.makedirs(self.config.result_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for repeat in range(1,self.config.repeat_cross_validate+1):
            repeat_checkpoint_folder = f'CV-{repeat}'
            os.makedirs(os.path.join(self.config.checkpoint_dir,repeat_checkpoint_folder), exist_ok=True)

            for fold, test_patients in enumerate(self.specify_fold_test_patient.values(), 1):
                if fold not in self.config.folds_to_train: 
                    print('-' * 15, f"Skip Fold {fold}", '-' * 15)
                    continue

            
                test_ids  = [f"Patient {id}" for id in test_patients]
                train_ids = [pid for pid in self.patient_ids if pid not in test_ids]

                print("Train Patients:", ','.join([d.split(' ')[1] for d in train_ids]))
                print("Test Patients:",  ','.join([d.split(' ')[1] for d in test_ids ]))


                train_dataloader, test_dataloader = self.__get__dataloader(train_ids,test_ids)
                model = self.__load__model(fold=f"fold-{fold}")

                self.finetune(model , 
                              train_dataloader, 
                              test_dataloader, 
                              ckpt_path = os.path.join(self.config.checkpoint_dir, repeat_checkpoint_folder, f"fold-{fold}-checkpoint.pth") )
        

    
    def finetune(self, model, train_dataloader, test_dataloader, ckpt_path = 'checkpoint.pth'):
        from torch import nn
        from torch import optim
        from tqdm import tqdm
        import transformers

        if self.config.continue_training and os.path.exists(ckpt_path):
            print(f"Loading Checkpoint from {ckpt_path}!")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt)

        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        best_model = None
        best_auc = 0.0


        train_metric , test_metric = Metric(), Metric()

        for epoch in range(self.config.num_epoch):
            ## Model Training
            model.train()
            for data in tqdm(train_dataloader):
                inputs, labels = data["clip"].to(self.device), data["label"].to(self.device).float().unsqueeze(-1)
                optimizer.zero_grad()

                if self.config.model == 'Diffusion':
                    outputs = model(data)

                else:
                    outputs = model(inputs)


                # handling huggingface output for training
                if isinstance(outputs, transformers.modeling_outputs.ImageClassifierOutput):
                    outputs = outputs.logits.float()
                

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                train_metric.update(y_true = labels.cpu().numpy(),y_pred = probs)


            ## Model Testing
            model.eval()
            with torch.no_grad():
                for data in tqdm(test_dataloader):
                    inputs = data["clip"].to(self.device)
                    if self.config.model == 'Diffusion':
                        outputs = model(data)

                    else:
                        outputs = model(inputs)

                    if isinstance(outputs, transformers.modeling_outputs.ImageClassifierOutput):
                        outputs = outputs.logits.float()

                    probs = torch.sigmoid(outputs).cpu().numpy()
                    test_metric.update(y_true = data["label"].numpy(),y_pred = probs)

            train_auc = train_metric.cal_auc_score()
            test_auc  = test_metric.cal_auc_score()
            print(f"Epoch:{epoch}, Train AUC: {train_auc} , Test AUC: {test_auc}")


            if test_auc > best_auc:
                best_auc = test_auc
                best_model = model.state_dict()
                torch.save(best_model,ckpt_path)

            train_metric.reset()
            test_metric.reset()

    
    def test_cross_validation(self):
        assert os.path.exists(self.config.checkpoint_dir), f"{self.config.checkpoint_dir} Does not exist!!"
        
        all_metric = []
        for repeat in range(1,self.config.repeat_cross_validate+1):
            repeat_checkpoint_folder = f'CV-{repeat}'
            repeat_metric = []

            for fold , test_patients in enumerate(self.specify_fold_test_patient.values() , 1):                            
            
                test_ids  = [f"Patient {id}" for id in test_patients]
                train_ids = [pid for pid in self.patient_ids if pid not in test_ids]

                print(f"\nInference on Fold {fold} Test Samples",f"Test Patient:{','.join(test_ids)}")
                
                train_dataloader, test_dataloader = self.__get__dataloader(train_ids, test_ids)
                model = self.__load__model(fold=f"fold-{fold}")

                eval_metric = self.inference(model = model, 
                                             test_dataloader = test_dataloader, 
                                             ckpt_path = os.path.join(self.config.checkpoint_dir, repeat_checkpoint_folder , f"fold-{fold}-checkpoint.pth" ))

                repeat_metric.append(eval_metric)
            
            all_metric.append(repeat_metric)
        
        return all_metric



    def inference(self, model , test_dataloader , ckpt_path):
        from tqdm import tqdm
        import transformers

        if os.path.exists(ckpt_path):
            print(f"Loading Checkpoint from {ckpt_path}!")
            if self.config.model == 'Diffusion':
                ckpt = torch.load(ckpt_path, map_location=self.device)
                new_state_dict = {k.replace("classifier_head.output_3d_layer.4.", "last."): v for k, v in ckpt.items()}
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(torch.load(ckpt_path))
        else:
            raise ValueError(f"{ckpt_path} does not exist @@")
        
        model.to(self.device)
        eval_metric = Metric(info=["patient_id","intervention"])
        model.eval()
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                inputs = data["clip"].to(self.device)

                if self.config.model == 'Diffusion':
                    outputs = model(data)

                else:
                    outputs = model(inputs)

                if isinstance(outputs, transformers.modeling_outputs.ImageClassifierOutput):
                    outputs = outputs.logits.float()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                eval_metric.update(y_true=data["label"].numpy(), y_pred=probs)

                info = {"patient_id":data["patient"], "intervention": data["intervention"]}
                eval_metric.update_info(info) # get the patient of each clip
                
        return eval_metric
    

        
    def get_fold_distribution(self):
        from collections import Counter

        fold_distribution = []

        for fold , test_ids in enumerate(self.specify_fold_test_patient.values() , 1):
            print(f"Fold-{fold} Test Distribution")
            test_ids  = [f"Patient {id}" for id in test_ids]
            train_ids = [pid for pid in self.patient_ids if pid not in test_ids]
            train_dl = self.__get__dataloader(train_ids , is_train=False)
            test_dl  = self.__get__dataloader(test_ids  , is_train=False)

            train_label = [label for _, label in train_dl.dataset]
            test_label  = [label for _, label in test_dl.dataset]

            train_counter = Counter(train_label)
            test_counter  = Counter(test_label)
            
            train_distribution = {"healthy":train_counter[0], "unhealthy":train_counter[1]}
            test_distribution  = {"healthy":test_counter[0],  "unhealthy":test_counter[1]}
            
            fold_distribution.append({"train":train_distribution,"test":test_distribution})
        
        return fold_distribution



        
