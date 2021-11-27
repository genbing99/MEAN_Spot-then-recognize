import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from numpy import argmax
from sklearn.metrics import confusion_matrix
import os

# For score aggregation, to smooth the spotting confidence score
def smooth(y, box_pts):
    y = [each_y[0] for each_y in y]
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def spotting(result, total_gt_spot, subject_count, p, metric_final, spot_multiple, k_p, final_samples, final_dataset_spotting):
    prev=0
    pred_subject = []
    gt_subject = []
    metric_video = MeanAveragePrecision2d(num_classes=1)
    for videoIndex, video in enumerate(final_samples[subject_count-1]):
        preds = []
        gt = []
        countVideo = len([video for subject in final_samples[:subject_count-1] for video in subject])
        print('Video:', countVideo+videoIndex)
        score_plot = np.array(result[prev:prev+len(final_dataset_spotting[countVideo+videoIndex])]) #Get related frames to each video
        score_plot = smooth(score_plot, k_p*2)
        score_plot_agg = score_plot.copy()
        #Plot the result to see the peaks
        plt.figure(figsize=(15,4))
        plt.plot(score_plot_agg) 
        plt.xlabel('Frame')
        plt.ylabel('Score')
        threshold = score_plot_agg.mean() + p * (max(score_plot_agg) - score_plot_agg.mean()) #Moilanen threshold technique
        if(spot_multiple):
            peaks, _ = find_peaks(score_plot_agg, height=threshold, distance=k_p)
            if(len(peaks)==0): #Occurs when no peak is detected, simply give a value to pass the exception in mean_average_precision
                preds.append([0, 0, 0, 0, 0, 0, 0]) 
            for peak in peaks:
                preds.append([peak-k_p, 0, peak+k_p, 0, 0, 0, peak]) #Extend left and right side of peak by k frames
        else:
            peak = np.where(score_plot_agg == max(score_plot_agg))[0][0]
            preds.append([peak-k_p, 0, peak+k_p, 0, 0, 0, peak])  
        for samples in video: 
            gt.append([samples[0], 0, samples[2], 0, 0, 0, 0, samples[1]])
            total_gt_spot += 1
            plt.axvline(x=samples[0], color='r')
            plt.axvline(x=samples[2]+1, color='r')
            plt.axhline(y=threshold, color='g')
        # plt.show()
        prev += len(final_dataset_spotting[countVideo+videoIndex])
        metric_video.add(np.array(preds),np.array(gt))
        metric_final.add(np.array(preds),np.array(gt)) #IoU = 0.5 according to MEGC2020 metrics
        pred_subject.append(preds)
        gt_subject.append(gt)
    return pred_subject, gt_subject, total_gt_spot, metric_video, metric_final

def confusionMatrix(gt, pred, show=False):
    TN_recog, FP_recog, FN_recog, TP_recog = confusion_matrix(gt, pred).ravel()
    f1_score = (2*TP_recog) / (2*TP_recog + FP_recog + FN_recog)
    num_samples = len([x for x in gt if x==1])
    average_recall = TP_recog / (TP_recog + FN_recog)
    average_precision = TP_recog / (TP_recog + FP_recog)
    return f1_score, average_recall, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, average_precision, average_recall

def history_plot(history_spot, history_recog, filename):
    f, ax = plt.subplots(1,3,figsize=(15,3)) 
    #Spot Loss vs Epochs
    ax[0].plot(history_spot.history['loss'])
    ax[0].plot(history_spot.history['val_loss'])
    ax[0].set_title('Spot Loss/Epochs')
    ax[0].set_ylabel('Spot Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['loss','val_loss'], loc='upper left')
    #Recog Loss vs Epochs
    ax[1].plot(history_recog.history['loss'])
    ax[1].plot(history_recog.history['val_loss'])
    ax[1].set_title('Recog Loss/Epochs')
    ax[1].set_ylabel('Recog Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['recog_loss, val_recog_loss'], loc='upper left')
    #Recog Accuracy vs Epochs
    ax[2].plot(history_recog.history['categorical_accuracy'])
    ax[2].plot(history_recog.history['val_categorical_accuracy'])
    ax[2].set_title('Recog Accuracy/Epochs')
    ax[2].set_ylabel('Recog Accuracy')
    ax[2].set_xlabel('Epoch')
    ax[2].legend(['recog_accuracy','val_recog_accuracy'], loc='upper left')
    # plt.show()

def sequence_evaluation(total_gt_spot, metric_final): #Get TP, FP, FN for final evaluation
    TP_spot = int(sum(metric_final.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP_spot = int(sum(metric_final.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN_spot = total_gt_spot - TP_spot
    print('TP:', TP_spot, 'FP:', FP_spot, 'FN:', FN_spot)
    return TP_spot, FP_spot, FN_spot

def convertLabel(dataset_name, emotion_class, label):
    if(dataset_name == 'CASME2'):
        label_dict = { 'disgust' : 0, 'happiness' : 1, 'others' : 2, 'surprise' : 3, 'repression' : 4 }
    elif((dataset_name == 'CASME_sq' or dataset_name == 'SAMMLV') and emotion_class == 4):
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3 }
    else:
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2 }
    return label_dict[label]
    
def splitVideo(y1_pred, subject_count, final_samples, final_dataset_spotting): #To split y1_act_test by video
    prev=0
    y1_pred_video = []
    for videoIndex, video in enumerate(final_samples[subject_count-1]):
        countVideo = len([video for subject in final_samples[:subject_count-1] for video in subject])
        y1_pred_each = y1_pred[prev:prev+len(final_dataset_spotting[countVideo+videoIndex])]
        y1_pred_video.append(y1_pred_each)
        prev += len(final_dataset_spotting[countVideo+videoIndex])
    return y1_pred_video

def recognition(dataset_name, emotion_class, result, preds, metric_video, final_emotions, subject_count, pred_list, gt_tp_list, y_test, final_samples, pred_window_list, pred_single_list, spot_multiple, k, k_p, final_dataset_spotting):
    cur_pred = []
    cur_tp_gt = []
    pred_gt_recog = []
    cur_pred_window = []
    cur_pred_single = []
    pred_emotion = splitVideo(result, subject_count, final_samples, final_dataset_spotting) #Split predicted emotion by video
    act_emotion = splitVideo(y_test, subject_count, final_samples, final_dataset_spotting)
    pred_match_gt = sorted(metric_video.value(iou_thresholds=0.5)[0.5][0]['pred_match_gt'].items())
    for video_index, video_match in pred_match_gt: #key=video_index, value=match index for each video
        for pred_index, sample_index in enumerate(video_match): #pred_index=index of prediction array, sample_index=index of emotion array
            try:
                pred_peak = max(0, preds[video_index][pred_index][-1]) #Last index is peak predicted
                # Case 1: Using peak only
                pred_emotion_list = argmax(pred_emotion[video_index][pred_peak], axis=-1)
#                 most_common_emotion, _ = Counter(pred_emotion_list).most_common(1)[0]
                cur_pred_single.append(pred_emotion_list)
                
                # Case 2: Using [peak-k_p, peak]
                pred_emotion_list = list(argmax(pred_emotion[video_index][max(0, pred_peak-k_p):max(1, pred_peak)], axis=-1))
                most_common_emotion, _ = Counter(pred_emotion_list).most_common(1)[0]
                cur_pred.append(most_common_emotion)
                
                # Case 3: Using [peak-k_p, peak+k_p]
                pred_emotion_list = list(argmax(pred_emotion[video_index][max(0, pred_peak-k_p):min(len(pred_emotion[video_index]),pred_peak+k_p)], axis=-1))
                most_common_emotion, _ = Counter(pred_emotion_list).most_common(1)[0]
                cur_pred_window.append(most_common_emotion)
                
                pred_gt_recog.append(argmax(pred_emotion[video_index][final_samples[subject_count-1][video_index][0][0]])) #Predicted emotion on gt onset label
                #For predicted tp only
                gt_label = final_emotions[subject_count-1][video_index][sample_index] #Get video emotion    
                if(sample_index!=-1):
                    cur_tp_gt.append(convertLabel(dataset_name, emotion_class, gt_label))
                else:
                    cur_tp_gt.append(-1)
            except Exception as e:
                print('Recognition Error:', e)
                pass
    pred_list.extend(cur_pred)
    gt_tp_list.extend(cur_tp_gt)
    pred_window_list.extend(cur_pred_window)
    pred_single_list.extend(cur_pred_single)
    # print('Predict on gt        :', pred_gt_recog)
    # print('Predicted with single:', cur_pred_single)
    # print('Predicted with window:', cur_pred_window)
    print('Predicted with k_p     :', cur_pred)
    return pred_list, gt_tp_list, pred_window_list, pred_single_list

def recognition_evaluation(dataset_name, emotion_class, final_gt, final_pred, show=False):
    if(dataset_name == 'CASME2'):
        label_dict = { 'disgust' : 0, 'happiness' : 1, 'others' : 2, 'surprise' : 3, 'repression' : 4 }
    elif((dataset_name == 'CASME_sq' or dataset_name == 'SAMMLV') and emotion_class == 4):
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3 }
    else:
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2 }
    
    #Display recognition result
    precision_list = []
    recall_list = []
    f1_list = []
    ar_list = []
    TP_all = 0
    FP_all = 0
    FN_all = 0
    TN_all = 0
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x==emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x==emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, precision_recog, recall_recog = confusionMatrix(gt_recog, pred_recog, show)
                if(show):
                    print(emotion.title(), 'Emotion:')
                    print('TP:', TP_recog, '| FP:', FP_recog, '| FN:', FN_recog, '| TN:', TN_recog)
#                     print('Total Samples:', num_samples, '| F1-score:', round(f1_recog, 4), '| Average Recall:', round(recall_recog, 4), '| Average Precision:', round(precision_recog, 4))
                TP_all += TP_recog
                FP_all += FP_recog
                FN_all += FN_recog
                TN_all += TN_recog
                precision_list.append(precision_recog)
                recall_list.append(recall_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        precision_list = [0 if np.isnan(x) else x for x in precision_list]
        recall_list = [0 if np.isnan(x) else x for x in recall_list]
        precision_all = np.mean(precision_list)
        recall_all = np.mean(recall_list)
        f1_all = (2 * precision_all * recall_all) / (precision_all + recall_all)
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        print('------ After adding ------')
        print('TP:', TP_all, 'FP:', FP_all, 'FN:', FN_all, 'TN:', TN_all)
        print('Precision:', round(precision_all, 4), 'Recall:', round(recall_all, 4))
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4), '| F1-Score:', round(f1_all, 4))
        return UF1, UAR
    except:
        return '', ''

#Evaluate mae and asr
def apex_evaluation(preds, gt, k_p):
    tp_apex = 0
    mae_total = 0
    for index in range(len(preds)):
        gt_onset = gt[index][0][0] #Onset
        gt_offset = gt[index][0][2]#Offset
        pred_apex = preds[index][0][0]+k_p #Predicted onset + k_p = predicted apex
        if(pred_apex > gt_onset and pred_apex < gt_offset):
            tp_apex += 1
        mae_total = mae_total + abs(pred_apex - gt[index][0][-1]) #Last index is the apex frame
    asr_score = tp_apex / len(preds)
    mae_score = mae_total / len(preds)
    return asr_score, mae_score

#  Create new directory for storing weights
def create_directory(train, dataset_name):
    path_main = 'MEAN_Weights'
    path_dataset = 'MEAN_Weights\\' + dataset_name
    path_spot = 'MEAN_Weights\\' + dataset_name + '\\' + 'spot'
    path_recog = 'MEAN_Weights\\' + dataset_name + '\\' + 'recog'
    path_spot_recog = 'MEAN_Weights\\' + dataset_name + '\\' + 'spot_recog'
    if train:
        if os.path.exists(path_main)==False:
            os.mkdir(path_main)
        if os.path.exists(path_dataset)==False:
            os.mkdir(path_dataset)
        if os.path.exists(path_spot)==False:
            os.mkdir(path_spot)
        if os.path.exists(path_recog)==False:
            os.mkdir(path_recog)
        if os.path.exists(path_spot_recog)==False:
            os.mkdir(path_spot_recog)