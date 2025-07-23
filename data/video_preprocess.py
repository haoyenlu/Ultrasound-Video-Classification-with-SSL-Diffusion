import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import os
from tqdm import tqdm
from ssim.ssimlib import SSIM, SSIMImage

from kneed import DataGenerator, KneeLocator

from data.utils import video_loader



class PreprocessCWSSIM:
    def __init__(self, 
                 video_dir, 
                 k=0.1,
                 width=10,
                 target_size=(224,224),
                 window_step=20,
                 num_frames=100,
                 new_video_dir = 'preprocess_videos'):
        """
        Remove Overlapping Frames using custom algorithm with CW-SSIM
        Args:
            video_dir: The videos that need preprocessing
            k, width: CW-SSIM parameter
            target_size: Resize the video to this size for comparison
            window_step, num_frames: Sliding-Window Approach for obtaining optimal threshold
            new_video_dir: The folder that stored the preprocessed videos
        """

        self.video_dir = video_dir
        self.patient_dir = os.listdir(self.video_dir)

        """CW-SSIM and Threshold Parameters"""
        self.k = k
        self.width = width
        self.target_size = target_size
        self.window_step = window_step
        self.num_frames = num_frames


        self.new_video_dir = new_video_dir
        os.makedirs(new_video_dir,exist_ok=True)

    
    def run(self):
        # get threshold
        print("Get Obtimal Trheshold using Kneedle Algorithm")
        threshold = self.obtain_optimal_threshold()

        for pid in self.patient_dir:
            print(f"CW-SSIM Preprocessed {pid}")
            os.makedirs(os.path.join(self.new_video_dir, pid),exist_ok=True)
            videos = os.listdir(os.path.join(self.new_video_dir, pid))

            for video in videos:
                ssim_filter_frames = self.cwssim_filter(os.path.join(self.video_dir,pid,video) , cwssim_threshold = threshold)

                H, W = ssim_filter_frames[0].size
                writer = cv2.VideoWriter(os.path.join(self.new_video_dir, pid ,video),cv2.VideoWriter_fourcc(*'XVID'), fps=12, frameSize=(H,W))
                for frame in ssim_filter_frames:
                    frame = np.array(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    writer.write(frame)
                writer.release()
        
        print("Finish CW-SSIM Preprocessing")


    def cwssim_curve_window(video_path, k =0.1, width = 30, target_size=(128,128) , window_step = 10, num_frames = 100):
        frames = video_loader(video_path)

        sim_index = []
        for j in range(0,len(frames) - num_frames , window_step):
            current_img = SSIM(img = frames[j], k=k , size=target_size)
        
            sim_index_window = []
            
            for i in tqdm(range(1,num_frames)):
                target_img = SSIMImage(frames[j+i],size=target_size)
                cwssim_val = current_img.cw_ssim_value(target_img, width=width)
                sim_index_window.append(cwssim_val)

            sim_index.append(sim_index_window)

        return sim_index




    def obtain_optimal_threshold(self):
        cwssim_data = []
        for pid in self.patient_dir:
            print(f"Processing {pid} videos")
            videos = os.listdir(os.path.join(self.video_dir, pid))
            for video in videos:
                data = self.cwssim_curve_window(os.path.join(self.video_dir, pid , video), k=self.k, width=self.width, target_size=self.target_size,window_step = self.window_step, num_frames = self.num_frames)
                cwssim_data.append(data)

        mean_cwssim = []
        for i in range(len(cwssim_data)):
            mean_cwssim.append(np.mean(cwssim_data[i],axis=0))

        mean_cwssim = np.mean(mean_cwssim,axis=0)

        x = list(range(0,mean_cwssim.shape[0]))
        y = mean_cwssim
        kneedle = KneeLocator(x, y, S= 0.1, curve="convex", direction="decreasing" , online=False)
        optimal_threshold = kneedle.knee_y

        return optimal_threshold
    
    def cwssim_filter(self, video_path, cwssim_threshold):
        frames = video_loader(video_path)
        new_frames = []

        current_img = SSIM(img = frames[0], k=self.k, size=self.target_size)
        current_img_index = 0
        new_frames.append(frames[0])

        pbar = tqdm(range(1,len(frames)))

        for i in pbar:
            target_img = SSIMImage(frames[i],size=self.target_size)
            cwssim_val = current_img.cw_ssim_value(target_img, width=self.width)

            # two images are not similar -> add the target image frame
            if cwssim_val <= cwssim_threshold: 
                new_frames.append(frames[i])
                current_img = SSIM(img = frames[i], k=self.k, size=self.target_size)
                current_img_index = i
            pbar.set_description(f"CW SSIM ({current_img_index}-{i}):{cwssim_val}")

        return new_frames
