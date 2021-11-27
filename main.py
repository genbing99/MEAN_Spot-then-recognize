import sys
import argparse
from face_crop import *
from load_images import *
from load_label import *
from load_excel import *
from feature_extraction import *
from prepare_training import *
from train_evaluate import *
from distutils.util import strtobool

## Note that the whole process will take a long time... please be patient
def main(config):

    # Define the dataset and expression to spot
    dataset_name = config.dataset_name
    train = config.train

    print(' ------ Spot-then-recognize', dataset_name, '-------')
    
    # Load Images
    print('\n ------ Face detection and Croping Images ------')
    face_crop(dataset_name) # Can comment this out after completed on the dataset specified and intend to try on another expression_type
    print("\n ------ Loading Images ------")
    images, subjects, subjectsVideos = load_images(dataset_name)
    
    # Load Ground Truth Label
    print('\n ------ Loading Excel ------')
    codeFinal = load_excel(dataset_name)
    print('\n ------ Loading Ground Truth From Excel ------')
    final_images, final_subjects, final_videos, final_samples, final_emotions = load_label(dataset_name, images, subjects, subjectsVideos, codeFinal) 

    # Set Parameters
    print('\n ------ Set k ------')
    k = set_k(dataset_name)
    print('\n ------ Set Emotion Class ------')
    emotion_class = set_emotion_class(dataset_name)
    print('\n ------ Set k_p ------') # k'
    k_p = cal_k_p(dataset_name, final_samples)

    # Feature Extraction & Pre-processing
    print('\n ------ Recognition Feature Extraction & Pre-processing ------')
    final_dataset_recognition = feature_extraction_recognition(dataset_name, final_images, final_samples)
    # pickle.dump(final_dataset_recognition, open(dataset_name + "_dataset_recognition.pkl", "wb")) # To save time when needed to run for several attempts
    # final_dataset_recognition = pickle.load( open( dataset_name + "_dataset_recognition.pkl", "rb" ) )
    
    print('\n ------ Spotting Feature Extraction & Pre-processing ------')
    final_dataset_spotting = feature_extraction_spotting(dataset_name, final_images, k)
    # pickle.dump(final_dataset_spotting, open(dataset_name + "_dataset_spotting.pkl", "wb")) # To save time when needed to run for several attempts
    # final_dataset_spotting = pickle.load( open( dataset_name + "_dataset_spotting.pkl", "rb" ) )

    # Spotting Pseudo-labeling
    print('\n ------ Spotting Pseudo-Labeling ------')
    pseudo_y = spotting_pseudo_labeling(dataset_name, final_samples, final_dataset_spotting, k_p)
    
    # Recognition labeling
    print('\n ------ Recognition Pseudo-Labeling ------')
    spot_multiple, X, y, X1, y1, X2, y2, emotion_list = recognition_label(dataset_name, emotion_class, final_samples, final_emotions, final_dataset_spotting, final_dataset_recognition, pseudo_y)
    
    # LOSO splitting
    print('\n ------ Leave one Subject Out ------')
    groupsLabel, groupsLabel1 = loso_split(X, y, X1, y1, X2, y2, final_subjects, final_samples, final_dataset_spotting, final_emotions, emotion_list)
    
    # Create directory if not exist
    create_directory(train, dataset_name)

    # Model Training & Evaluation
    print('\n ------ MEAN Training & Testing ------')
    TP_spot, FP_spot, FN_spot, metric_final, gt_list, pred_list, gt_tp_list, asr_score, mae_score = train_test(X, y, X1, y1, X2, y2, dataset_name, emotion_class, groupsLabel, groupsLabel1, spot_multiple, final_subjects, final_emotions, final_samples, final_dataset_spotting, k, k_p, 'micro-expression', epochs_spot=10, epochs_recog=100, spot_lr=0.0005, recog_lr=0.0005, batch_size=32, ratio=5, p=0.55, spot_attempt=1, recog_attempt=1, train=train)
    
    # Model Final Evaluation
    print('\n ------ MEAN Final Evaluation ------')
    final_evaluation(TP_spot, FP_spot, FN_spot, dataset_name, 'micro-expression', metric_final, asr_score, mae_score, spot_multiple, pred_list, gt_list, emotion_class, gt_tp_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--dataset_name', type=str, default='CASME2') 
    parser.add_argument('--train', type=strtobool, default=False) #Train or use pre-trained weight for prediction
    
    config = parser.parse_args()

    main(config)