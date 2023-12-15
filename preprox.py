import os

#os.chdir("/drive/MyDrive/data")


import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import utils


# Directory where mp3 are stored
AUDIO_DIR = '/home/diego/fma/data'

# Load metadata and features.
tracks = utils.load(AUDIO_DIR + '/fma_metadata/tracks.csv')

#Select the desired subset among the entire dataset
sub = 'small'
subset = tracks[tracks['set', 'subset'] <= sub]

#Part of subset dataset containing information about top_genres


#definition of labels and track_ids
labels = np.array(subset['track']['genre_top'])

print("labels shape: ", labels.shape)


#Function to convert labels into array of probabilities
def conv_label(label,number_of_classes=8):
  le = LabelEncoder()
  y = le.fit_transform(label)
  y = to_categorical(y, number_of_classes)
  return y


#Here we define functions for the creation of useful datasets for CNNs learning

#Function for the creation of 1d audio signal dataset
def splitAudio(subset, data_type):
    print("splitting audio...")
    # Creare un array vuoto
    a = np.array([0,0])
    split = subset[subset['set', 'split'] == data_type]
    labels = conv_label(split['track']['genre_top'])    
    for i in range(len(split.index)): 
        try:          
            audio = np.array(getAudio(split.index[i])[0])
            y = np.array(labels[i])
            new_row = np.array([audio, y], dtype=object)
            a = np.vstack([a, new_row])
        except:
            print('problems with song: ', split.index[i])
    return a[1:]




#Functions for the creation of .npy files for heavy arrays (split arrays in multiple files and then reconstrict it)

def saveheavy2(a, name, n):  #functions to save heavy arrays in multiple files
    print("saving heavy array...")
    l= len(a)
    if(n>1):
        for i in range(n-1):
            a_i = a[int((l/n)*i):int((l/n)*(i+1))]
            np.save(f'{name}_{i+1}.npy', a_i)
    a_i = a[int((l/n)*(n-1)):l]
    np.save(f'{name}_{n}.npy', a_i)
    
def readheavy2(name, n, Dir):
    a = np.array([0,0])
    for i in range(n):
        new_a = np.load(f'data/{Dir}/{name}_{i+1}.npy', allow_pickle = True)
        a = np.vstack([a, new_a])
    return a[1:]  

#Creation of Audio dataframe 

test = splitAudio(subset, 'test')
training = splitAudio(subset, 'training')
validation = splitAudio(subset, 'validation')


#Save heavy arrays
#saveheavy2(test, 'test', 10)
#saveheavy2(training, 'training', 10)
saveheavy2(validation, 'validation', 2)