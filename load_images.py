import glob
import natsort
import numpy as np
import pandas as pd
import cv2
import pickle

def load_images(dataset_name):
  images = []
  subjects = []
  subjectsVideos = []

  print('Loading images from dataset', dataset_name)

  # For dataset CASME_sq
  if(dataset_name == 'CASME_sq'):
      for i, dir_sub in enumerate(natsort.natsorted(glob.glob(dataset_name + "\\rawpic_crop\\*"))):
        print('Subject: ' + dir_sub.split('\\')[-1])
        subjects.append(dir_sub.split('\\')[-1])
        subjectsVideos.append([])
        for dir_sub_vid in natsort.natsorted(glob.glob(dir_sub + "\\*")):
          subjectsVideos[-1].append(dir_sub_vid.split('\\')[-1].split('_')[1][:4]) # Ex:'CASME_sq/rawpic_aligned/s15/15_0101disgustingteeth' -> '0101' 
          image = []
          for dir_sub_vid_img in natsort.natsorted(glob.glob(dir_sub_vid + "\\img*.jpg")):
            image.append(cv2.imread(dir_sub_vid_img, 0))
          print('Done -> ' + dir_sub_vid.split('\\')[-1])
          images.append(np.array(image))
      
  # For dataset SAMMLV
  elif(dataset_name == 'SAMMLV'):
      for i, dir_vid in enumerate(natsort.natsorted(glob.glob(dataset_name + "\\SAMM_longvideos_crop\\*"))):
        subject = dir_vid.split('\\')[-1].split('_')[0]
        if (subject not in subjects): #Only append unique subject name
          subjects.append(subject)
          subjectsVideos.append([])
        subjectsVideos[-1].append(dir_vid.split('\\')[-1])

        image = []
        for dir_vid_img in natsort.natsorted(glob.glob(dir_vid + "\\*.jpg")):
          image.append(cv2.imread(dir_vid_img, 0))
        image = np.array(image)
        print('Done -> ' + dir_vid.split('\\')[-1])
        images.append(image)
      
  # For dataset CASME2
  elif(dataset_name == 'CASME2'):
      #Read only frames from onset->offset for recognition
      for i, dir_sub in enumerate(natsort.natsorted(glob.glob(dataset_name + "\\CASME2\\CASME2-RAW_crop\\sub*"))):
        subject_name = dir_sub.split('\\')[-1][3:].lstrip('0')
        print('Subject: ' + subject_name)
        subjects.append(subject_name)
        subjectsVideos.append([])
        for dir_sub_vid in natsort.natsorted(glob.glob(dir_sub + "\\EP*")):
          video_name = dir_sub_vid.split('\\')[-1].split('EP')[1]
          subjectsVideos[-1].append(video_name) # Ex: 'CASME2\sub01\EP02_01f'->'02_01f'
          image = []
          for dir_sub_vid_img in natsort.natsorted(glob.glob(dir_sub_vid + "\\*.jpg")):
              image.append(cv2.imread(dir_sub_vid_img, 0))
          images.append(np.array(image))
          print('Done -> ' + dir_sub_vid.split('\\')[-1])

  # For dataset SMIC_HS/SMIC_VIS/SMIC_NIR
  elif('SMIC' in dataset_name):
      video_type = dataset_name.split('_')[1]
      for i, dir_sub in enumerate(natsort.natsorted(glob.glob("SMIC\\SMIC_E_" + video_type + "_crop\\s*"))):
        print('Subject: ' + dir_sub.split('\\')[-1])
        subjects.append(dir_sub.split('\\')[-1])
        subjectsVideos.append([])
        for dir_sub_vid in natsort.natsorted(glob.glob(dir_sub + "\\s*")):
          subjectVideosName = dir_sub_vid.split('\\')[-1].split('_')[1] + '_' + dir_sub_vid.split('\\')[-1].split('_')[2]
          subjectsVideos[-1].append(subjectVideosName) #ne_01, po_01
          image = []
          for dir_sub_vid_img in natsort.natsorted(glob.glob(dir_sub_vid + "\\*.jpg")):
              image.append(cv2.imread(dir_sub_vid_img, 0))
          print('Done -> ' + dir_sub_vid.split('\\')[-1])
          images.append(np.array(image))

  print('Loading images from dataset', dataset_name, 'All Done')
  return images, subjects, subjectsVideos

# Save data into pkl
def save_images_pkl(dataset_name, images, subjectsVideos, subjects):
    pickle.dump(images, open(dataset_name + "_images_crop.pkl", "wb") )
    pickle.dump(subjectsVideos, open(dataset_name + "_subjectsVideos_crop.pkl", "wb") )
    pickle.dump(subjects, open(dataset_name + "_subjects_crop.pkl", "wb") )

# Load data into pkl
def load_images_pkl(dataset_name):
    images = pickle.load( open( dataset_name + "_images_crop.pkl", "rb" ) )
    subjectsVideos = pickle.load( open( dataset_name + "_subjectsVideos_crop.pkl", "rb" ) )
    subjects = pickle.load( open( dataset_name + "_subjects_crop.pkl", "rb" ) )
    return images, subjectsVideos, subjects

if __name__ == "__main__":
    load_images('CASME2')