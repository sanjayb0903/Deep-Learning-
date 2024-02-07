#!/usr/bin/env python3
import os
import librosa
import math
import json

DATASET_PATH = '<path_to_the_dataset>'
SAMPLE_RATE = 22050
DURATION = 30 #in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION   #22048*30=661500
JSON_PATH = 'json_file_path'                 #data_10.json

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):   
    
   #dictionary to store data
    data = {
        "mapping": [],    
        "labels": [],     
        "mfcc": []        
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments) #66150 samples per segment
    
    

    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)   
    
    #loop through all the folders and genres
    for i, (dirpath, dirnames, filenames ) in enumerate(os.walk(dataset_path)):

       

        #ensure that we are not at the root level 
        if dirpath is not dataset_path:  
        

            #save semantic labels
            dirpath_components = dirpath.split("/")  
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            #process all audio files
            for f in filenames:

                #load the audio file
                filepath = os.path.join(dirpath,f)
                signal, sr = librosa.load(filepath, sr=SAMPLE_RATE)



                #process segments extracting mfcc and storing data
                for s in range (num_segments):

                    #calculate start and end sample
                    start_sample = num_samples_per_segment * s       
                    finish_sample = start_sample + num_samples_per_segment 


                    #extract mfcc feature
                    mfcc = librosa.feature.mfcc(signal[start_sample: finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length = hop_length)
                    mfcc = mfcc.T

                    #store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["label"].append(i-1)
                        print("{},segment:{}".format(filepath, s+1))

        # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    print("Starting..")
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10) 
