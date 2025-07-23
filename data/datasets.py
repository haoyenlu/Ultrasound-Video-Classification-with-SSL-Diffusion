from torch.utils.data import Dataset
import torch
import os
import copy
from PIL import Image
import numpy as np

from data.utils import video_loader



def make_dataset_sliding_window(video_path, labels = None, num_frames = 16 , step_size = 16,
                                sex_feature = None, age_feature = None, ppt_feature = None):   
    '''Create video clip dataset from list of videos'''
    dataset = []

    for i, video in enumerate(video_path):
        video_frames = video_loader(video)
        label = labels[i] if labels is not None else None
        sex = sex_feature[i] if sex_feature is not None else None
        age = age_feature[i] if age_feature is not None else None
        ppt = ppt_feature[i] if ppt_feature is not None else None
        
        sample = {
            'video': video,
            'n_frames': num_frames,
            'label': label,
            'sex': sex,
            'age': age,
            'ppt': ppt,
        }

        for j in range(0, (len(video_frames) - num_frames), step_size):
            sample_i = copy.deepcopy(sample)
            sample_i['frame_indices'] = list(range(j, j + num_frames))
            dataset.append(sample_i)
    

    return dataset






class VideoDataset(Dataset):
    '''Generate the dataset using the video'''
    def __init__(self, video_path, labels,
                 sex_feature = None, age_feature = None,
                 ppt_feature = None,
                 spatial_transform=None, temporal_transform=None, augmentation = None,
                 num_frames=16, 
                 step_size = 4,
                 channel_first = True,):
        """
        Args:
            video_path: root dir of the videos
            spatial_transform: transformation for each frame
            temporal_transform: transformation for timestep
            sample_duration: number of frames for each data
        """
        
        assert len(video_path) == len(labels), "Number of Videos should be the same as the labels"

        self.video_path = video_path

        self.data = make_dataset_sliding_window(
            video_path, labels, num_frames, step_size,
            sex_feature=sex_feature, age_feature=age_feature,
            ppt_feature=ppt_feature
        )

        self.num_frames = num_frames

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.augmentation = augmentation
        self.loader = video_loader
        self.channel_first = channel_first


    def _image_to_tensor(self, img):

        img = img.astype(np.uint8)

        if len(img.shape) == 2:  # add channel dimension to gray scale image
            img = torch.from_numpy(np.expand_dims(img, axis = 0))
        elif len(img.shape) == 3: # convert (H , W , C) to (C , H , W)
            img = torch.from_numpy(img.transpose((2, 0, 1))) 

        # normalize
        img = img / 255.0

        # backward compatibility
        return img.float()

    def _to_tensor(self, data):
        data = np.array(data)
        data = torch.from_numpy(data).float()
        return data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            frames: the target frame
            labels: the target frame label
        """
        path = self.data[index]['video']
        label = self.data[index]['label']
        frame_indices = self.data[index]['frame_indices']
        frames = self.loader(path) # load all the frame of the video

        clip = [frames[i] for i in frame_indices]   # get only the clip of the videos

        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip] # output shape: (T , C, H , W)

        if self.augmentation is not None:
            clip = self.augmentation(clip)


        clip = [self._image_to_tensor(img) for img in clip]
        clip = torch.stack(clip,0)

        if self.channel_first:
            clip = clip.permute(1, 0, 2, 3) # (C, T , H , W)
        
        patient = os.path.basename(path).split('_')[0]
        intervention = os.path.basename(path).split('_')[1]
        sex = self._to_tensor(self.data[index]['sex'])
        age = self._to_tensor(self.data[index]['age'])
        ppt = self._to_tensor(self.data[index]['ppt'])

        data = {"clip" : clip, "label": label, "patient":patient , "intervention": intervention, 
                "sex": sex, "age": age, "ppt": ppt}

        return data
    

    def __len__(self):
        return len(self.data)
    



class VideoFrameDataset(Dataset):
    def __init__(self,
                 files,
                 transforms = None):
        
        super().__init__()

        self.files = files
        self.transforms = transforms



    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = self.transforms(img) if self.transforms is not None else img
        return img
    
    def __len__(self):
        return len(self.files)
    