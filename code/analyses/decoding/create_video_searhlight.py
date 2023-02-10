#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:32:55 2022

@author: yair
"""
import os
import glob
import cv2

fn_patterns = ['number_subject_*_50_50_5.0',
               'he_she_*_50_50_5',
               'he_they_*_50_50_5', 
               'embedding_vs_long_3rd_word_*_50_50_5',
               'dec_quest_len2_*_50_50_5',
               'unacc_unerg_dec_*_50_50_5']

fig_type = 'inflate_True'
fn_patterns = ['number_subject_*_50_50_5.0_5.0']
block_trains = ['auditory', 'visual']
block_tests = ['auditory', 'visual']
alpha = 0.001
tmin, tmax = 0.1, 0.6

for block_train in block_trains:
    for block_test in block_tests:
        for fn_pattern in fn_patterns:


            image_folder = '../../../Figures/viz_brain/'
            video_name = f'{fn_pattern}_{block_train}_{block_test}_query_pval_min_{alpha}_mean_t_*_{fig_type}.avi'
            video_name = os.path.join(image_folder, video_name)
            
            fn_pattern = f'{fn_pattern}_{block_train}_{block_test}_query_pval_min_{alpha}_mean_t_*_{fig_type}.png'
            fn_pattern = os.path.join(image_folder, fn_pattern)
            images = sorted(glob.glob(fn_pattern))
            
            
            if len(images) == 0:
                continue
            
            frame = cv2.imread(images[0])
            height, width, layers = frame.shape
            
            fps = len(images)/(tmax-tmin)/10
            video = cv2.VideoWriter(video_name, 0, fps, (width,height))
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            for image in images:
                im = cv2.imread(image)
                ws = image.split('_')
                IX_t = ws.index('t')
                t = ws[IX_t+1]
                cv2.putText(im, f't={t}', (10,23), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                video.write(im)
                
            
            cv2.destroyAllWindows()
            video.release()
            print(f'Video saved: {video_name}')