import numpy as np
import pandas as pd
import cv2
import dlib
import time

def pol2cart(rho, phi): #Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def computeStrain(u, v): #Compute os , setting t=1 to maximize the sensitivity of ME
    u_x = u - pd.DataFrame(u).shift(1, axis=1)
    v_y = v - pd.DataFrame(v).shift(1, axis=0)
    u_y = u - pd.DataFrame(u).shift(1, axis=0)
    v_x = v - pd.DataFrame(v).shift(1, axis=1)
    os = np.array(np.sqrt((u_x**2).fillna(0) + (v_y**2).fillna(0) + 1/2 * (u_y.fillna(0)+v_x.fillna(0))**2))
    return os

def preProcess(img1, img2, shape):
    # ROI 1 (Left Eyebrow)
    x31=max(shape.part(17).x - 12, 0) #3
    y32=max(shape.part(19).y - 12, 0)
    x33=min(shape.part(21).x + 12, 128)
    y34=min(shape.part(41).y + 12, 128)

    # ROI 2 (Right Eyebrow)
    x41=max(shape.part(22).x - 12, 0) # 3
    y42=max(shape.part(24).y - 12, 0)
    x43=min(shape.part(26).x + 12, 128)
    y44=min(shape.part(46).y + 12, 128)

    # ROI 3 #Mouth
    x51=max(shape.part(60).x - 12, 0) # 5
    y52=max(shape.part(50).y - 12, 0)
    x53=min(shape.part(64).x + 12, 128)
    y54=min(shape.part(57).y + 12, 128)

    # Compute Optical Flow Features
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(img1, img2, None)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
    u, v = pol2cart(magnitude, angle)
    os = computeStrain(u, v)
    
    # Features Concatenation into 128x128x3
    final = np.zeros((128, 128, 3))
    final[:,:,0] = u
    final[:,:,1] = v
    final[:,:,2] = os
    
    # Normalize the image
    for channel in range(3):
        final[:,:,channel] = cv2.normalize(final[:,:,channel], None, alpha=0, beta=1,norm_type=cv2.NORM_MINMAX)    
    
    # ROI Selection -> Image resampling into 42x22x3
    final_image = np.zeros((42, 42, 3))
    final_image[:21, :, :] = cv2.resize(final[min(y32, y42) : max(y34, y44), x31:x43, :], (42, 21))
    final_image[21:42, :, :] = cv2.resize(final[y52:y54, x51:x53, :], (42, 21))
        
    return final_image

def feature_extraction_recognition(dataset_name, final_images, final_samples):
    # Get dlib landmark detection file
    predictor_model = "Utils\\shape_predictor_68_face_landmarks.dat"
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    final_videos_samples = [videos for subjects in final_samples for videos in subjects]
    final_samples = [samples for subjects in final_samples for videos in subjects for samples in videos]

    print('Running')
    start = time.time()
    dataset = []

    rect = dlib.rectangle(0,0,128,128) # For dlib landmark detection
    if dataset_name == 'CASME_sq' or dataset_name == 'SAMMLV':
        sample_count = 0
        for video in range(len(final_images)):
            ref_img = final_images[video][0]
            shape = face_pose_predictor(ref_img,rect)
            # Only onset and apex
            for sample in final_videos_samples[video]:
                onset = sample[0]
                apex = sample[1]
                img1 = final_images[video][onset]
                img2 = final_images[video][apex]
                final_image = preProcess(img1, img2, shape)
                dataset.append(final_image)
                print('Video:', video, 'Done')
                sample_count+=1

    elif dataset_name == 'CASME2':
        # Only onset and apex
        for video in range(len(final_images)):
            ref_img = final_images[video][0]
            shape = face_pose_predictor(ref_img,rect)
            onset = final_samples[video][0]
            apex = final_samples[video][1]
            img1 = final_images[video][onset]
            img2 = final_images[video][apex]
            final_image = preProcess(img1, img2, shape)
            dataset.append(final_image)
            print('Video', video, 'Done')
        
    elif 'SMIC' in dataset_name:
        for video in range(len(final_images)):
            ref_img = final_images[video][0]
            shape = face_pose_predictor(ref_img,rect)
            onset = final_samples[video][0]
            offset = final_samples[video][2]
            img1 = final_images[video][onset]
            max_dif = 0

            # Loop from onset until offset to find maximum difference
            for count_k in range(offset-onset): 
                img2 = final_images[video][onset+count_k] 
                opt_image = preProcess(img1, img2, shape)
                frame_dif = sum(sum(sum(opt_image)))
                if(max_dif < frame_dif):
                    max_dif = frame_dif
                    final_image = opt_image
                    final_samples[video][1] = onset+count_k # Set the apex frame in the samples, Ex. [0, -1, 35] -> [0, 13, 35]
            dataset.append(final_image)   
            print('Video', video, 'Done')

    print('All Done')
    end = time.time()
    print('Total time taken: ' + str(end-start) + 's')
    return dataset

def feature_extraction_spotting(dataset_name, final_images, k):
    # Get dlib landmark detection file
    predictor_model = "Utils\\shape_predictor_68_face_landmarks.dat"
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    print('Running')
    start = time.time()
    dataset = []
    for video in range(len(final_images)):
        OFF_video = []
        ref_img = final_images[video][0]
        rect = dlib.rectangle(0,0,128,128)
        shape = face_pose_predictor(ref_img,rect)
        
        # Use sliding window [F_i, F_i+k] to extract optical flow features
        for img_count in range(final_images[video].shape[0]-k):
            img1 = final_images[video][img_count]
            img2 = final_images[video][img_count+k]
            final_image = preProcess(img1, img2, shape) 
            OFF_video.append(final_image)
        dataset.append(OFF_video)   
        print('Video', video, 'Done')

    print('All Done')
    end = time.time()
    print('Total time taken: ' + str(end-start) + 's')
    return dataset