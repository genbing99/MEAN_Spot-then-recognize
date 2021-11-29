import os
import shutil
import glob
import natsort
import dlib
import cv2

face_detector = dlib.get_frontal_face_detector()

def face_crop(dataset_name):
    print('Running Face Detection & Face Cropping for', dataset_name)

    # For dataset CASME_sq
    if(dataset_name == 'CASME_sq'):

        # Create new directory for 'rawpic_crop'
        dir_crop = dataset_name + '\\rawpic_crop\\'
        if os.path.exists(dir_crop)==False:
            os.mkdir(dir_crop)

        # Read all images from 'rawpic' then preprocess into 'rawpic_crop'
        for subjectName in glob.glob(dataset_name + '\\rawpic\\*'):
            dataset_rawpic = dataset_name + '\\rawpic\\' + str(subjectName.split('\\')[-1]) + '\\*'
            print('Running Subject', subjectName.split('\\')[-1])

            # Create new directory for each subject in 'rawpic_crop'
            dir_crop_sub = dataset_name + '\\rawpic_crop\\' + str(subjectName.split('\\')[-1]) + '\\'
            if os.path.exists(dir_crop_sub):
                shutil.rmtree(dir_crop_sub)
            os.mkdir(dir_crop_sub)

            # Read videos from 'rawpic'
            for vid in glob.glob(dataset_rawpic):

                # Create new directory for each video in 'rawpic_crop'
                dir_crop_sub_vid = dir_crop_sub + vid.split('\\')[-1] 
                if os.path.exists(dir_crop_sub_vid): 
                    shutil.rmtree(dir_crop_sub_vid)
                os.mkdir(dir_crop_sub_vid)

                # Read images from 'raw_pic'
                for dir_crop_sub_vid_img in natsort.natsorted(glob.glob(vid+'\\img*.jpg')): 
                    img = dir_crop_sub_vid_img.split('\\')[-1]

                    # Get img num Ex 001,002,...,2021
                    count = img[3:-4] 
                    # Load the image
                    image = cv2.imread(dir_crop_sub_vid_img)
                    # Dlib face detection
                    detected_faces = face_detector(image, 1)

                    # Use first frame as reference frame for face cropping
                    if (count == '001'): 
                        for face_rect in detected_faces:
                            face_top = face_rect.top()
                            face_bottom = face_rect.bottom()
                            face_left = face_rect.left()
                            face_right = face_rect.right()

                    # Crop and resize to 128x128
                    face = image[face_top:face_bottom, face_left:face_right] 
                    face = cv2.resize(face, (128, 128)) 

                    # Write image into 'rawpic_crop'
                    cv2.imwrite(dir_crop_sub_vid + "\\img{}.jpg".format(count), face)

                print('Done Video', vid.split('\\')[-1])

    # For dataset SAMM Long Videos
    elif(dataset_name == 'SAMMLV'):

        # Create new directory for 'SAMM_longvideos_crop'
        if os.path.exists(dataset_name + '\\SAMM_longvideos_crop'): 
            shutil.rmtree(dataset_name + '\\SAMM_longvideos_crop')
        os.mkdir(dataset_name + '\\SAMM_longvideos_crop')

        # Read all images from 'SAMM_longvideos' then preprocess into 'SAMM_longvideos_crop'
        for vid in glob.glob(dataset_name + '\\SAMM_longvideos\\*'):
            count = 0
            dir_crop = dataset_name + '\\SAMM_longvideos_crop\\' + vid.split('\\')[-1]

            # Create new directory for each video in 'SAMM_longvideos_crop'
            if os.path.exists(dir_crop): 
                shutil.rmtree(dir_crop)
            os.mkdir(dir_crop)
            print('Video', vid.split('\\')[-1])

            # Read videos from 'SAMM_longvideos'
            for img_count, dir_crop_img in enumerate(natsort.natsorted(glob.glob(vid+'\\*.jpg'))):
                img = dir_crop_img.split('\\')[-1].split('.')[0]

                # Get img num Ex 0001,0002,...,2021
                count = img.split('_')[-1]
                # Load the image
                image = cv2.imread(dir_crop_img)
                # Dlib face detection
                detected_faces = face_detector(image, 1)

                # Use first frame as reference frame for face cropping
                if (img_count == 0):
                    for i, face_rect in enumerate(detected_faces):
                        face_top = face_rect.top()
                        face_bottom = face_rect.bottom()
                        face_left = face_rect.left()
                        face_right = face_rect.right()

                # Crop and resize to 128x128
                face = image[face_top:face_bottom, face_left:face_right]
                face = cv2.resize(face, (128, 128)) 

                # Write image into 'rawpic_crop'
                cv2.imwrite(dir_crop + "\\{}.jpg".format(count), face)

            print('Done Video', vid.split('\\')[-1])

    # For dataset SMIC_HS/SMIC_VIS/SMIC_NIR, *Note there might be slightly difference in file structure obtained from original authors
    if('SMIC' in dataset_name):

        # Get video type, HS/VIS/NIR
        video_type = dataset_name.split('_')[1]
        dataset_type = dataset_name.split('_')[0]

        # Create new directory for 'SMIC_E_{video_type}_crop'
        dir_crop = dataset_type + '\\SMIC-E_raw image\\' + video_type + '\\'
        if os.path.exists(dataset_type + '\\SMIC_E_' + video_type + '_crop'):
            shutil.rmtree(dataset_type + '\\SMIC_E_' + video_type + '_crop')
        os.mkdir(dataset_type + '\\SMIC_E_' + video_type + '_crop')
        
        # Read all images from 'SMIC-E_raw image\\{video_type}_long\\SMIC-{video_type}-E\\{video_type}' image then preprocess into 'SMIC_E_{video_type}_crop'
        for subjectName in natsort.natsorted(glob.glob(dataset_type + '\\SMIC-E_raw image\\' + video_type + '_long\\SMIC-' + video_type + '-E\\' + video_type + '\\*')):
            dataset_rawpic = dataset_type + '\\SMIC-E_raw image\\' + video_type + '_long\\SMIC-' + video_type + '-E\\' + video_type + '\\' + str(subjectName.split('\\')[-1]) + '\\*' 
            
            # Create new directory for each subject in 'SMIC_E_{video_type}_crop'
            dir_crop_sub = dataset_type + '\\SMIC_E_' + video_type + '_crop\\' + str(subjectName.split('\\')[-1]) + '\\' # Ex: SMIC\SMIC_E_HS_crop\s01
            if(str(subjectName.split('\\')[-1])[0] == 's'): # Prevent from creating directory for .xlsx files
                if os.path.exists(dir_crop_sub):
                    shutil.rmtree(dir_crop_sub)
                os.mkdir(dir_crop_sub)
                print('Subject', subjectName.split('\\')[-1])

            # Read videos from 'SMIC-E_raw image\\{video_type}_long\\SMIC-{video_type}-E\\{video_type}'
            for vid in natsort.natsorted(glob.glob(dataset_rawpic)):

                # Create new directory for each video in 'SMIC-E_raw image\\{video_type}_long\\SMIC-{video_type}-E\\{video_type}'
                dir_final_vid = dir_crop_sub + vid.split('\\')[-1] + '\\' # Ex: SMIC\SMIC_E_HS_crop\s01\s1_ne_01
                if os.path.exists(dir_final_vid): 
                    shutil.rmtree(dir_final_vid)
                os.mkdir(dir_final_vid)
                print('Video', vid.split('\\')[-1])

                # Read images from 'SMIC-E_raw image\\{video_type}_long\\SMIC-{video_type}-E\\{video_type}'
                for img_count, dir_final_img in enumerate(natsort.natsorted(glob.glob(vid+'\\*.jpg'))):
                    img = dir_final_img.split('\\')[-1]

                    # Conditions to get img num Ex 559700, 559701..., please change according to file structure
                    if(video_type == 'HS'): 
                        img_index = img.split('.')[0].split('image')[1] 
                    else:
                        if(' ' in img):
                            img_index = img.split('.')[0].split(' ')[1]
                        else:
                            img_index = img

                    # Load the image
                    image = cv2.imread(dir_final_img)
                    # Dlib face detection
                    detected_faces = face_detector(image, 1)
                    # Use first frame as reference frame for face cropping
                    if (img_count == 0):
                        for face_rect in detected_faces:
                            face_top = face_rect.top()
                            face_bottom = face_rect.bottom()
                            face_left = face_rect.left()
                            face_right = face_rect.right()

                    # Crop and resize to 128x128
                    face = image[face_top:face_bottom, face_left:face_right] 
                    face = cv2.resize(face, (128, 128)) 

                    # Write image into 'SMIC_E_{video_type}_crop'
                    cv2.imwrite(dir_final_vid + "\\image{}.jpg".format(img_index), face)

                print('Done Video', vid.split('\\')[-1])

    # For dataset CASME2
    if(dataset_name == 'CASME2'):

        # Create new directory for 'CASME2-RAW_crop'
        dir_crop = dataset_name + '\\CASME2\\CASME2-RAW_crop\\'
        if os.path.exists(dataset_name + '\\CASME2\\CASME2-RAW_crop'):
            shutil.rmtree(dataset_name + '\\CASME2\\CASME2-RAW_crop')
        os.mkdir(dataset_name + '\\CASME2\\CASME2-RAW_crop')

        # Read all images from 'CASME2-RAW' then preprocess into 'CASME2-RAW_crop'
        for subjectName in glob.glob(dataset_name + '\\CASME2\\CASME2-RAW\\*'):
            dataset_rawpic = dataset_name + '\\CASME2\\CASME2-RAW\\' + str(subjectName.split('\\')[-1]) + '\\*'

            # Create new directory for each subject in 'CASME2-RAW_crop'
            dir_crop_sub = dataset_name + '\\CASME2\\CASME2-RAW_crop\\' + str(subjectName.split('\\')[-1]) + '\\'
            if os.path.exists(dir_crop_sub):
                shutil.rmtree(dir_crop_sub)
            os.mkdir(dir_crop_sub)
            print('Subject', subjectName.split('\\')[-1])

            # Read videos from 'CASME2-RAW'
            for vid in glob.glob(dataset_rawpic):

                # Create new directory for each video in 'CASME2-RAW_crop'
                dir_crop_sub_vid = dir_crop_sub + vid.split('\\')[-1] 
                if os.path.exists(dir_crop_sub_vid): 
                    shutil.rmtree(dir_crop_sub_vid)
                os.mkdir(dir_crop_sub_vid)
                print('Video', vid.split('\\')[-1])

                # Read images from 'CASME2-RAW'
                for dir_crop_sub_vid_img in natsort.natsorted(glob.glob(vid+'\\img*.jpg')):
                    img = dir_crop_sub_vid_img.split('\\')[-1]

                    # Get img num Ex 001,002,...,290
                    img_index = img.split('.')[0].split('img')[1] 
                    # Load the image
                    image = cv2.imread(dir_crop_sub_vid_img)
                    # Dlib face detection
                    detected_faces = face_detector(image, 1)

                    # Use first frame as reference frame for face cropping
                    if (int(img_index) == 1):
                        for face_rect in detected_faces:
                            face_top = face_rect.top()
                            face_bottom = face_rect.bottom()
                            face_left = face_rect.left()
                            face_right = face_rect.right()

                    # Crop and resize to 128x128
                    face = image[face_top:face_bottom, face_left:face_right]
                    face = cv2.resize(face, (128, 128)) 

                    # Write image into 'CASME2-RAW_crop'
                    cv2.imwrite(dir_crop_sub_vid + "\\img{}.jpg".format(img_index), face)
                print('Done Video', vid.split('\\')[-1])

    print('Face Detection and Face Cropping for', dataset_name, 'All Done!!')

if __name__ == "__main__":
    # face_crop('CASME2')
    face_crop('SMIC_HS') # Change dir
    face_crop('SMIC_NIR')
    face_crop('SMIC_VIS')