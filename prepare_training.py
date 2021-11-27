import numpy as np
from tensorflow.keras.utils import to_categorical

def spotting_pseudo_labeling(dataset_name, final_samples, final_dataset_spotting, k_p):
    pseudo_y = [] #Pseudolabel spotting [0,1]
    video_count = 0 

    for subject_index, subject in enumerate(final_samples):
        for video_index, video in enumerate(subject):
            samples_arr = []
            if (len(video)==0):
                pseudo_y.append([0 for i in range(len(final_dataset_spotting[video_count]))]) #Last k_p frames are ignored
            else:
                pseudo_y_each = [0]*(len(final_dataset_spotting[video_count]))
                for ME_index, ME in enumerate(video):
                    samples_arr.append(np.arange(ME[0], ME[2]+1))
                for arr_index, ground_truth_arr in enumerate(samples_arr): 
                    for index in range(len(pseudo_y_each)):
                        pseudo_arr = np.arange(index, index+k_p) 
                        # Heaviside step function, if IoU>0 then y=1, else y=0
                        if (pseudo_y_each[index] < len(np.intersect1d(pseudo_arr, ground_truth_arr))/len(np.union1d(pseudo_arr, ground_truth_arr))):
                            pseudo_y_each[index] = 1 
                pseudo_y.append(pseudo_y_each)
            video_count+=1
            
    # Integrate all videos into one dataset
    pseudo_y = [y for x in pseudo_y for y in x]
    print(dataset_name, 'Total frames:', len(pseudo_y))
    return pseudo_y

def recognition_label(dataset_name, emotion_class, final_samples, final_emotions, final_dataset_spotting, final_dataset_recognition, pseudo_y):
    # Detect only the highest peak or all peaks above threshold p
    if(dataset_name == 'CASME_sq' or dataset_name == 'SAMMLV'): 
        spot_multiple = True
    else:
        spot_multiple = False
        
    # Determine the emotions used for evaluation
    if (dataset_name == 'CASME_sq' or dataset_name == 'SAMMLV'):
        emotion_list = ['negative', 'positive', 'surprise']
    else:
        emotion_list = ['repression', 'anger', 'contempt', 'disgust', 'fear', 'sadness', 'negative', 'happiness', 'positive', 'surprise', 'others', 'other']

    X1 = [] # Recognition validation
    y1 = [] # Recognition validation 
    X2 = [] # Recognition training
    y2 = [] # Recognition training
    label_videos = [videos for subjects in final_samples for videos in subjects] # Get [onset, apex, offset] for each video
    emotion_videos = [videos for subjects in final_emotions for videos in subjects] # Get emotion for each video

    video_count = 0
    for video_index in range(len(final_dataset_spotting)): 
        for sample_index in range(len(label_videos[video_index])):
            if(emotion_videos[video_index][sample_index] in emotion_list):
                image_index = label_videos[video_index][sample_index][0]
                X1.append(final_dataset_spotting[video_index][image_index])
                y1.append(emotion_videos[video_index][sample_index])
                X2.append(final_dataset_recognition[video_count])
            video_count+=1
        
    y1 = np.array(y1) 
    #Pseudolabel Recognition
    if (dataset_name == 'CASME2'): # Convert 0, 1, 2 to negative, positive, surprise
        y1 = [0 if ele=='disgust' else ele for ele in y1]
        y1 = [1 if ele=='happiness' else ele for ele in y1]
        y1 = [2 if ele=='others' else ele for ele in y1]
        y1 = [3 if ele=='surprise' else ele for ele in y1]
        y1 = [4 if ele=='repression' else ele for ele in y1]
    else: # Convert 0, 1, 2 to negative, positive, surprise
        y1 = [0 if ele=='negative' else ele for ele in y1]
        y1 = [1 if ele=='positive' else ele for ele in y1]
        y1 = [2 if ele=='surprise' else ele for ele in y1]

    all_images = [frame for video in final_dataset_spotting for frame in video]
    X = features_split(all_images) # For spotting training
    y = np.array(pseudo_y) # For spotting training
    X1 = features_split(X1) # For recognition validation
    y1 = to_categorical(y1)  # For recognition validation
    X2 = features_split(X2) # For recognition training
    y2 = y1 # For recognition training
    return spot_multiple, X, y, X1, y1, X2, y2, emotion_list

#Split the features for channel-wise learning
def features_split(X):
    X_copy = X.copy()
    for index in range(len(X_copy)):
        u = np.array(X_copy[index][:,:,0].reshape(42,42,1))
        v = np.array(X_copy[index][:,:,1].reshape(42,42,1))
        os = np.array(X_copy[index][:,:,2].reshape(42,42,1))
        X_copy[index] = [u, v, os]
    return X_copy

def loso_split(X, y, X1, y1, X2, y2, final_subjects, final_samples, final_dataset_spotting, final_emotions, emotion_list):
    #To split the dataset by subjects
    groupsLabel = y.copy() # For spotting
    groupsLabel1 = [] # For recognition
    prevIndex = 0
    countVideos = 0
    videos_len = []

    #Get total frames of each video
    for video_index in range(len(final_dataset_spotting)):
        videos_len.append(len(final_dataset_spotting[video_index]))

    print('Frame Index for each subject (Spotting):-')
    for subject_index in range(len(final_samples)):
        countVideos += len(final_samples[subject_index])
        index = sum(videos_len[:countVideos])
        groupsLabel[prevIndex:index] = subject_index
        print('Subject', final_subjects[subject_index], ':', prevIndex, '->', index)
        prevIndex = index

    #Get total frames of each video
    print('\nFrame Index for each subject (Recognition):-')
    for subject_index in range(len(final_samples)):
        for video_index in range(len(final_samples[subject_index])):
            for sample_index in range(len(final_samples[subject_index][video_index])):
                if(final_emotions[subject_index][video_index][sample_index] in emotion_list):
                    groupsLabel1.append(subject_index)
        if(subject_index in np.unique(groupsLabel1)):
            print('Subject', final_subjects[subject_index], ':', len(groupsLabel1)-len(final_samples[subject_index]), '->', len(groupsLabel1)-1)
        else:
            print('Subject', final_subjects[subject_index], ':', 'Not available')

    print('\nTotal X:', len(X), '| Total y:', len(y), '| Total X1:', len(X1), '| Total y1:', len(y1), '| Total X2:', len(X2), '| Total y2:', len(y2))
    return groupsLabel, groupsLabel1