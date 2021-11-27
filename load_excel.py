import pandas as pd

def load_excel(dataset_name):

    # For dataset CASME_sq
    if(dataset_name == 'CASME_sq'):
        xl = pd.ExcelFile(dataset_name + '/code_final.xlsx') #Specify directory of excel file
        colsName = ['subject', 'video', 'onset', 'apex', 'offset', 'au', 'emotion', 'type', 'selfReport']
        codeFinal = xl.parse(xl.sheet_names[0], header=None, names=colsName) #Get data

        videoNames = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(videoName.split('_')[0])
        codeFinal['videoName'] = videoNames
        naming1 = xl.parse(xl.sheet_names[2], header=None, converters={0: str})
        dictVideoName = dict(zip(naming1.iloc[:,1], naming1.iloc[:,0]))
        codeFinal['videoCode'] = [dictVideoName[i] for i in codeFinal['videoName']]
        naming2 = xl.parse(xl.sheet_names[1], header=None)
        dictSubject = dict(zip(naming2.iloc[:,2], naming2.iloc[:,1]))
        codeFinal['subjectCode'] = [dictSubject[i] for i in codeFinal['subject']]
        
    # For dataset SAMMLV
    elif(dataset_name=='SAMMLV'):
        xl_SAMMLV = pd.ExcelFile(dataset_name + '/SAMM_LongVideos_V2_Release.xlsx')
        xl_SAMM = pd.ExcelFile('SAMM/SAMM_20181215_Micro_FACS_Codes_v2.xlsx')
        colsName_SAMMLV = ['Subject', 'Filename', 'Inducement Code', 'Onset', 'Apex', 'Offset', 'Duration', 'Type', 'Action Units', 'Notes']
        codeFinal_SAMMLV = xl_SAMMLV.parse(xl_SAMMLV.sheet_names[0], header=None, names=colsName_SAMMLV, skiprows=[0,1,2,3,4,5,6,7,8,9])
        colsName_SAMM = ['Subject', 'Filename', 'Inducement Code', 'Onset', 'Apex', 'Offset', 'Duration', 'Type', 'Action Units', 'Emotion', 'Classes', 'Notes']
        codeFinal_SAMM = xl_SAMM.parse(xl_SAMM.sheet_names[0], header=None, names=colsName_SAMM, skiprows=[0])
        codeFinal_SAMM['Filename'] = codeFinal_SAMM['Filename'].astype('object')
        
        codeFinal = codeFinal_SAMMLV.merge(codeFinal_SAMM[['Filename', 'Type', 'Emotion']], on=['Filename', 'Type']) #Merge two excel files
        videoNames = []
        subjectName = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(str(videoName).split('_')[0] + '_' + str(videoName).split('_')[1])
            subjectName.append(str(videoName).split('_')[0])
        codeFinal['videoCode'] = videoNames
        codeFinal['subjectCode'] = subjectName
        codeFinal['Type'].replace({'Micro - 1/2': 'micro-expression'}, inplace=True)
        codeFinal.rename(columns={'Type':'type', 'Onset':'onset', 'Offset':'offset', 'Apex':'apex', 'Emotion':'emotion'}, inplace=True) 

    # For dataset CASME2
    elif(dataset_name == 'CASME2'):
        xl = pd.ExcelFile(dataset_name + '/CASME2/CASME2_label_Ver_2.xls') #Specify directory of excel file
        codeFinal = xl.parse(xl.sheet_names[0]) #Get data
        codeFinal.rename(columns={
            'Subject':'subject', 
            'Filename':'video', 
            'OnsetFrame':'onset', 
            'ApexFrame':'apex', 
            'OffsetFrame':'offset',
            'Action Units':'au', 
            'Estimated Emotion':'emotion'}, inplace=True)
        videoNames = []
        subjectNames = []
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(videoName.split('EP')[1])
        codeFinal['videoName'] = videoNames #Redundant
        codeFinal['videoCode'] = videoNames
        codeFinal['type'] = 'micro-expression'
        codeFinal['subjectCode'] = codeFinal['subject'] #Redundant
        
    # For dataset SMIC_HS/SMIC_VIS/SMIC_NIR
    if('SMIC' in dataset_name): #SMIC does not have excel file provided
        video_type = dataset_name.split('_')[1]
        xl = pd.ExcelFile('SMIC/SMIC-E_raw image/' + video_type + '_long/SMIC-' + video_type + '-E_annotation.xlsx') #Specify directory of excel file
        
        codeFinal = xl.parse(xl.sheet_names[0]) #Get data
        #NIR got different onset and offset
        codeFinal.rename(columns={
            'Subject':'subject', 
            'Filename':'video', 
            'OnsetF':'onset', 
            'OffsetF':'offset',
            'Emotion':'emotion',
            'FirstF': 'firstFrame'}, inplace=True)
        videoNames = []
        subjectNames = []
        codeFinal = codeFinal[:157] #Remove last few rows contain of text
        for videoName in codeFinal.iloc[:,1]:
            videoNames.append(videoName.split('_')[1] + '_' + videoName.split('_')[2])
        for subjectName in codeFinal['subject']:
            subjectNames.append('s' + str(subjectName).zfill(2))
        codeFinal['onset'] = codeFinal['onset'] - codeFinal['firstFrame']
        codeFinal['offset'] = codeFinal['offset'] - codeFinal['firstFrame']
        codeFinal['apex'] = 0 #Apex frame not provided
        codeFinal['videoName'] = videoNames #Redundant
        codeFinal['videoCode'] = videoNames
        codeFinal['subjectCode'] = subjectNames
        codeFinal['type'] = 'micro-expression'

    print(codeFinal.columns)
    return codeFinal

if __name__ == "__main__":
    load_excel('CASME2')
