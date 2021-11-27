import tensorflow.compat.v1 as tf
import os
from tensorflow import keras
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from numpy import argmax
from sklearn.metrics import accuracy_score
import time

from training_utils import *
from define_model import *
random.seed(1)
tf.disable_v2_behavior()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def train_test(X, y, X1, y1, X2, y2, dataset_name, emotion_class, groupsLabel, groupsLabel1, spot_multiple, final_subjects, final_emotions, final_samples, final_dataset_spotting, k, k_p, expression_type, epochs_spot=10,  epochs_recog=100, spot_lr=0.0005, recog_lr=0.0005, batch_size=32, ratio=5, p=0.55, spot_attempt=1, recog_attempt=1, train=False):
    start = time.time()
    loso = LeaveOneGroupOut()
    subject_count = 0
    total_gt_spot = 0
    metric_final = MeanAveragePrecision2d(num_classes=1)
    adam_spot = keras.optimizers.Adam(lr=spot_lr)
    adam_recog = keras.optimizers.Adam(lr=recog_lr)
    model_spot = MEAN_Spot(adam_spot)
    weight_reset_spot = model_spot.get_weights() #Initial weights
    
    print(train, '---------------------------')

    # For Spotting
    gt_spot_list = []
    pred_spot_list = []
    # For recognition
    gt_list = []
    pred_list = []
    gt_tp_list = []
    pred_ori_list = []
    pred_window_list = []
    pred_single_list = []
    asr_score = 0
    # For LOSO
    spot_train_index = []
    spot_test_index = []
    recog_train_index = []
    recog_test_index = []
    
    recog_subject_uni = np.unique(groupsLabel1) # Get unique subject label
    for train_index, test_index in loso.split(X, y, groupsLabel): # Spotting Leave One Subject Out
        spot_train_index.append(train_index)
        spot_test_index.append(test_index)
    for subject_index, (train_index, test_index) in enumerate(loso.split(X1, y1, groupsLabel1)): # Recognition Leave One Subject Out
        if (subject_index not in recog_subject_uni): # To remove subject that don't have chosen emotions for evalaution 
            recog_train_index.append(np.array([]))
            recog_test_index.append(np.array([]))
        recog_train_index.append(train_index)
        recog_test_index.append(test_index)
        
    for subject_index in range(len(final_subjects)):
        subject_count+=1
        print('Index: ' + str(subject_count-1) + ' | Subject : ' + str(final_subjects[subject_count-1]))

        # Prepare training & testing data by loso splitting
        X_train, X_test = [X[i] for i in spot_train_index[subject_index]], [X[i] for i in spot_test_index[subject_index]] #Get training set spotting
        y_train, y_test = [y[i] for i in spot_train_index[subject_index]], [y[i] for i in spot_test_index[subject_index]] #Get testing set spotting
        X1_train, X1_test = [X1[i] for i in recog_train_index[subject_index]], [X1[i] for i in recog_test_index[subject_index]] #Get training set recognition
        y1_train, y1_test = [y1[i] for i in recog_train_index[subject_index]], [y1[i] for i in recog_test_index[subject_index]] #Get testing set recognition
        X2_train, y2_train = [X2[i] for i in recog_train_index[subject_index]], [y2[i] for i in recog_train_index[subject_index]] #Get training set recognition
        X2_test, y2_test = [X2[i] for i in recog_test_index[subject_index]], [y2[i] for i in recog_test_index[subject_index]] #Get testing set recognition

        print('Dataset Labels (Spotting, Recognition)', Counter(y_train), Counter([argmax(i) for i in y1_train]))
        
        # Make the dataset in the expected ratio by randomly removing training samples
        unique, uni_count = np.unique(y_train, return_counts=True) 
        rem_count = int(uni_count.min() * ratio) 
        if(rem_count<=len(y_train)):
            rem_index = random.sample([index for index, i in enumerate(y_train) if i==0], rem_count)
            rem_index += (index for index, i in enumerate(y_train) if i==1)
        else:
            rem_count = int(uni_count.max() / ratio)
            rem_index = random.sample([index for index, i in enumerate(y_train) if i==1], rem_count) 
            rem_index += (index for index, i in enumerate(y_train) if i==0)
        rem_index.sort()
            
        X_train = [X_train[i] for i in rem_index]
        y_train = [y_train[i] for i in rem_index]
        
        print('After Downsampling (Spotting, Recognition)', Counter(y_train), Counter([argmax(i) for i in y1_train]))
        
        print('------ MEAN Spotting-------') #To reset the model at every LOSO testing
        path = 'MEAN_Weights\\' + dataset_name + '\\' + 'spot'+ '\\s' + str(subject_count) + '.hdf5'
        
        #Prepare training & testing data
        X_train, X_test = [np.array(X_train)[:,0],np.array(X_train)[:,1], np.array(X_train)[:,2]], [np.array(X_test)[:,0],np.array(X_test)[:,1], np.array(X_test)[:,2]]
        
        if not train: # Load Pretrained Weights
            model_spot.load_weights(path)  
        else: 
            model_spot.set_weights(weight_reset_spot) 
            history_spot = model_spot.fit(
                X_train, np.array(y_train),
                batch_size=batch_size,
                epochs=epochs_spot,
                verbose=0,
                validation_data = (X_test, np.array(y_test)),
                shuffle=True,
                callbacks=[keras.callbacks.ModelCheckpoint(
                    filepath = path,
                    save_weights_only=True
                )],
            )
        
        path = 'MEAN_Weights\\' + dataset_name + '\\' + 'recog' + '\\s' + str(subject_count) + '.hdf5'
        model_recog = MEAN_Recog_TL(model_spot, adam_recog, emotion_class)
        
        if(len(X1_train) > 0): # Check the subject has samples for recognition
            print('------ MEAN Recognition-------') # Using transfer learning for recognition
            #Prepare training & testing data
            X1_train, X1_test = [np.array(X1_train)[:,0],np.array(X1_train)[:,1], np.array(X1_train)[:,2]], [np.array(X1_test)[:,0],np.array(X1_test)[:,1], np.array(X1_test)[:,2]]
            X2_train, X2_test = [np.array(X2_train)[:,0],np.array(X2_train)[:,1], np.array(X2_train)[:,2]], [np.array(X2_test)[:,0],np.array(X2_test)[:,1], np.array(X2_test)[:,2]]

            if not train: # Load Pretrained Weights
                model_recog.load_weights(path)  
            else: # Reset weights to ensure the model does not have info about current subject
                history_recog = model_recog.fit(
                    X2_train, np.array(y2_train),
                    batch_size=batch_size,
                    epochs=epochs_recog,
                    verbose=0,
                    validation_data = (X1_test, np.array(y1_test)),
                    shuffle=True,
                    callbacks=[keras.callbacks.ModelCheckpoint(
                        filepath = path,
                        save_weights_only=True
                    )],
                )
            result_recog_ori = model_recog.predict(
                X2_test, 
                verbose=1
            )

            if train: # Plot graph to see performance
                history_plot(history_spot, history_recog, str(final_subjects[subject_count-1]))

        path = 'MEAN_Weights\\' + dataset_name + '\\' + 'spot_recog' + '\\s' + str(subject_count) + '.hdf5'
        model_spot_recog = MEAN_Spot_Recog_TL(model_spot, model_recog, adam_recog)

        if train:
            model_spot_recog.save_weights(path) # Save Weights
        else:
            model_spot_recog.load_weights(path) # Load Pretrained Weights
        results = model_spot_recog.predict(
            X_test,
            verbose=1
        )
        
        print('---- Spotting Results ----')
        preds, gt, total_gt_spot, metric_video, metric_final = spotting(results[0], total_gt_spot, subject_count, p, metric_final, spot_multiple, k_p, final_samples, final_dataset_spotting)
        TP_spot, FP_spot, FN_spot = sequence_evaluation(total_gt_spot, metric_final)
        try:
            precision = TP_spot/(TP_spot+FP_spot)
            recall = TP_spot/(TP_spot+FN_spot)
            F1_score = (2 * precision * recall) / (precision + recall)
            # print('F1-Score = ', round(F1_score, 4))
            # print("COCO AP@[.5:.95]:", round(metric_final.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))
        except:
            pass
        pred_spot_list.extend(preds)
        gt_spot_list.extend(gt)
        asr_score, mae_score = apex_evaluation(pred_spot_list, gt_spot_list, k_p)
        # print('ASR:', round(asr_score,4))
        # print('MAE:', round(mae_score,4))
            
        if(len(X1_train) > 0): # Check the subject has samples for recognition
            #Recognition  
            print('---- Recognition Results ----')
            gt_list.extend(list(argmax(y2_test, -1)))
            pred_ori_list.extend(list(argmax(result_recog_ori, -1)))
            pred_list, gt_tp_list, pred_window_list, pred_single_list = recognition(dataset_name, emotion_class, results[1], preds, metric_video, final_emotions, subject_count, pred_list, gt_tp_list, y_test, final_samples, pred_window_list, pred_single_list, spot_multiple, k, k_p, final_dataset_spotting)
            print('Ground Truth           :', list(argmax(y2_test, -1)))
            
    print('Done Index: ' + str(subject_count-1) + ' | Subject : ' + str(final_subjects[subject_count-1]))

    end = time.time()
    print('Total time taken for training & testing: ' + str(end-start) + 's')
    return TP_spot, FP_spot, FN_spot, metric_final, gt_list, pred_list, gt_tp_list, asr_score, mae_score

def final_evaluation(TP_spot, FP_spot, FN_spot, dataset_name, expression_type, metric_final, asr_score, mae_score, spot_multiple, pred_list, gt_list, emotion_class, gt_tp_list):
    #Spotting
    precision = TP_spot/(TP_spot+FP_spot)
    recall = TP_spot/(TP_spot+FN_spot)
    F1_score = (2 * precision * recall) / (precision + recall)
    print('----Spotting----')
    print('Final Result for', dataset_name)
    print('TP:', TP_spot, 'FP:', FP_spot, 'FN:', FN_spot)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    print("COCO AP@[.5:.95]:", round(metric_final.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))
    print('ASR = ', round(asr_score, 4))
    print('MAE = ', round(mae_score, 4))

    #Check recognition accuracy if only correctly predicted spotting are considered
    if(not spot_multiple):
        print('\n----Recognition (All)----')
        print('Predicted    :', pred_list)
        print('Ground Truth :', gt_list)
        UF1, UAR = recognition_evaluation(dataset_name, emotion_class, gt_list, pred_list, show=True)
        print('Accuracy Score:', round(accuracy_score(gt_list, pred_list), 4))

    print('\n----Recognition (Consider TP only)----')
    gt_tp_spot = []
    pred_tp_spot = []
    for index in range(len(gt_tp_list)):
        if(gt_tp_list[index]!=-1):
            gt_tp_spot.append(gt_tp_list[index])
            pred_tp_spot.append(pred_list[index])
    print('Predicted    :', pred_tp_spot)
    print('Ground Truth :', gt_tp_spot)
    UF1, UAR = recognition_evaluation(dataset_name, emotion_class, gt_tp_spot, pred_tp_spot, show=True)
    print('Accuracy Score:', round(accuracy_score(gt_tp_spot, pred_tp_spot), 4))