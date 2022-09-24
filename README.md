# Sound Classification Using Deep Learning

This repository contains different projects on sound classification.
## Urban Sound Classification
The Urban Sound Classification Folder is a classification project for the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset.

This dataset contains **8732 labeled sound excerpts** (<=4seconds) of urban sounds from **10 classes**: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The classes are drawn from the urban sound taxonomy.

The file consists of a audio file and metadata file.
#### The Audio File 
8732 audio files of urban sounds in WAV format. The sampling rate, bit depth, and number of channels are the same as those of the original file uploaded to Freesound (and hence may vary from file to file).

#### The Metadata file 
This file contains meta-data information about every audio file in the dataset. The details maybe found [here](https://urbansounddataset.weebly.com/urbansound8k.html)

### Preprocessing 
The preprocessing jupyter EDA (Exploratory Data Analysis) is done where we analyze different tracks in the dataset and find ways to process it and develop a model.

### Model Development
The audio_classifier dataset the code is divided into 6 parts:
    
    Uploading the Data
        
        1. We used a opendatasets library from jovian to load the dataset from the source. 
        2. We have unzipped the tar file and extracted all the contents of the file.
        3. We removed the file named '.DS_Store' in each of the dataset folder which is threat to the system.
    
    Data Preprocessing
    
    1. It is the process of conversion of raw data into numerical features theat can be processed while preserving the original information.
    2. Since the data is audio which is collected in the analog form it should be converted into digital form and then analyzed for features.

    1. We saw in the EDA part that each signal had different
        1. Bit_depth
        2. Sample Rate 
        3. Coverting everything to mono.
    2. We can use librosa librbary which will help us overcome all the above points.For much of the preprocessing we will be able to use Librosa’s load() function, which by default converts the sampling rate to 22.05 KHz, normalise the data so the bit-depth values range between -1 and 1 and flattens the audio channels into mono.
## Feature Extraction 
    3. Now we have to extract the features. We have to convert them into visual representation which will allow us to indentify features for classification.
    For doing this there are popularly 2 methods:
        1. MFCC -  Mel-Frequency Cepstral Coefficients  
        2. Spectrograms

    Spectrograms are a useful technique for visualising the spectrum of frequencies of a sound and how they vary during a very short period of time.

    But spectrograms does not take into consideration the quality of the same sound. So we make use mfcc which are much more sensitive and here mfcc uses quasi-logarithmic spaced frequency scale, which is more similar to how the human auditory system processes sounds.

    For each audio file in the dataset, we will extract an MFCC (meaning we have an image representation for each audio sample) and store it in a Panda Dataframe along with it’s classification label. For this we will use Librosa’s mfcc() function which generates an MFCC from time series audio data.

    We save the values extracted into a CSV file.

    Load the Data

        We load the data from the csv file. It consists of 3 columns. Features
            It consists of the mfcc values. 
        Class
            The index value of the class that the mfcc value belongs to.
        Fold_component
            The folder value the data is taken from.
    We convert the index of classes to its class names that the tracks belong to.

    Model Creation
    
    We split the data into X_train and X_test. We develop a sequential neural network with 2 hidden layers each with 100 units and a dropout value of 0.1, with relu as activation and an ouput layer with softmax as activation.

        Why Softmax?
        It performs better for functions with multiclass classification problem.

        Why Dropout of 0.1?
        To prevent overfitting from happening dropout layer is used.
        Adam optimizer is used and the model is measured for accuracy with a loss fuction of categorical crossentropy since its a multiclass classification.
    We are saving the model values as a hdf5 files in the saved_models folder. This can be used later to predict files and shared with others.
    Prediction

    We predict on different track files to get the output.
    
    Saving the Model and Reusing the Weights.
    I have defined fuctions to print the output of the class it belongs to and also how to use the save weights and predict the classes that the track belongs to.
