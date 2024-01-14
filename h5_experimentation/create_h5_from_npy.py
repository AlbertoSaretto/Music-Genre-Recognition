import numpy as np
import h5py


"""
# Create h5 files to store dataset from npy files

This script creates train, valid, test h5 files from many .npy files. These files contain arrays of mp3. 
This works if the dataset is saved in many .npy files called training_1.npy, training_2.npy, validation_1.npy, validation_2.npy, test_1.npy, test_2.npy etc.

## Arguments:

create: bool = True if you want to create a new h5 file, False adds to existing h5 file
set_type: str = training, validation, test
filename_npy=f"../data/audio_array/{set_type}_{i}.npy"

## Usage:
- If you want to create a file from scratch:
    create=True,
    set_type="train" / "valid" / "test"
    filename_npy: the numpy file you want to load first. In my case: "../data/audio_array/training_{i}.npy"
                    Leave the {i}. The i is set when calling the script (see next point).

    How to run the script:  

        - Go to bash terminal and run: python class_CreateTrain 1
            where the number "1" is the i argument in the filename_npy, as said before.

- If you already have an h5 file created as before and you want to add new files to it:

    create = False

    The rest is equal to previous point (creating from scratch).

    In case of many files (for example I have 16 training_{i}.npy files), it is useful to use a bash script that
    automatically calls this function many times and loops over i.
    In my case I create manually the first file and then I use bash to loop from 2 to 16.


## Final outcome:
The output are two files called {set_type}_x and {set_type}_y (eg train_x.h5, train_y.h5). 
Inside the files there is one dataset whose key is "{set_type}" (eg ["train"])
You can access data with the following:

self.x = h5py.File(f"{set_type}_x.h5"),"r")[f"{set_type}"] (see DataAudioH5)
eg self.x = h5py.File(train_x.h5"),"r")["train"]

It's important to notice that the name of the file is identical to the key used to access data inside of it.
        
"""



class CreateTrain():
    def __init__(self, create=False, set_type="train", filename_npy="../data/audio_array/training_2.npy"):
        self.create = create
        self.set_type = set_type
        self.filename_npy = filename_npy



    def all_same_dimension(self, array_clips):
        indices_to_delete = []

        for i, audio in enumerate(array_clips[:, 0]):
            if audio.shape != (660984,):
                audio = audio[:660984]
                array_clips[i, 0] = audio

            if audio.shape != (660984,):
                # This happens when the audio is too short
                print(f"Deleting {i}")
                indices_to_delete.append(i)

        # Delete the rows after the loop
        array_clips = np.delete(array_clips, indices_to_delete, 0)

        print(f"Deleted {len(indices_to_delete)}")
        return array_clips

    # Rest of your code remains unchanged...

    def create_first(self, filename_h5="train_x.h5", filename_npy="../data/audio_array/training_1.npy"):
        train = np.load(filename_npy, allow_pickle=True)
        train_reshaped = self.all_same_dimension(train)
        stacked = np.stack(train_reshaped[:, 0])

        print("Following should be 660984", stacked.shape[1])

        # Create an HDF5 file and write the arrays to it
        with h5py.File(filename_h5, 'w') as hf:
            hf.create_dataset(f"{self.set_type}", data=stacked, maxshape=(None, stacked.shape[1]),compression="gzip", compression_opts=9)

    
    def create_first_labels(self, filename_h5="train_y.h5", filename_npy="../data/audio_array/training_1.npy"):
        train = np.load(filename_npy, allow_pickle=True)
        y = np.stack(train[:, 1])
        print(y.shape)

        with h5py.File(filename_h5, 'w') as hf:
            hf.create_dataset(f'{self.set_type}', data=y, maxshape=(None, y.shape[0]),compression="gzip", compression_opts=9)


    def add_to_h5(self, filename_h5="train_x.h5", filename_npy="../data/audio_array/training_2.npy"):
        train = np.load(filename_npy, allow_pickle=True)
        train = self.all_same_dimension(train)
        stacked = np.stack(train[:, 0])

        print("Following should be 660984", stacked.shape[1])

        # Create an HDF5 file and write the arrays to it
        with h5py.File(filename_h5, 'a') as hf:
            hf[f"{self.set_type}"].resize((hf[f"{self.set_type}"].shape[0] + stacked.shape[0]), axis=0)
            hf[f"{self.set_type}"][-stacked.shape[0]:] = stacked

    def add_to_h5_labels(self, filename_h5="train_y.h5", filename_npy="../data/audio_array/training_2.npy"):
        train = np.load(filename_npy, allow_pickle=True)
        y = np.stack(train[:, 1])
        print(y.shape)

        # Create an HDF5 file and write the arrays to it
        with h5py.File(filename_h5, 'a') as hf:
            hf[f"{self.set_type}"].resize((hf[f"{self.set_type}"].shape[0] + y.shape[0]), axis=0)
            hf[f"{self.set_type}"][-y.shape[0]:] = y


def main(i):
    print("i è " , i)
    
    ct = CreateTrain(create=False, set_type="test",filename_npy=f"../data/audio_array/test_{i}.npy")
   
    if ct.create:
            print("creating from scratch")
            ct.create_first(filename_h5=f"{ct.set_type}_x.h5", filename_npy=ct.filename_npy)
            ct.create_first_labels(filename_h5=f"{ct.set_type}_y.h5", filename_npy=ct.filename_npy)

    else:
        print("Adding to existing h5 file")
        ct.add_to_h5(filename_h5=f"{ct.set_type}_x.h5", filename_npy=ct.filename_npy)
        ct.add_to_h5_labels(filename_h5=f"{ct.set_type}_y.h5", filename_npy=ct.filename_npy)
            

if __name__ == "__main__":
    import sys
    # This lets you call the script from the command line
    # python script_name.py i
    # where i is the number of the file you want to load (eg 2 for training_2.npy)
    i = int(sys.argv[1])
    main(i)


