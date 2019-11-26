import pandas as pd
from os.path import exists
num_subjects = 15
subjects = [ 'subject{:02d}'.format(i) for i in range(1, num_subjects+1) ]
glasses_present = ['glasses', 'noglasses']
light_positions = ['leftlight', 'rightlight', 'centerlight']
emotions = ['happy', 'normal', 'sad', 'sleepy', 'surprised', 'wink']
directory = './dataset/compressed/'
features = [ 'filename' ]
features.extend(subjects)
features.extend(glasses_present)
features.extend(light_positions)
features.extend(emotions)
def newSample( subject ):
    sample = { key:0 for key in features }
    sample[subject] = 1
    if subject == 'subject08' or subject == 'subject13':
        sample[ 'glasses' ] = 1
    else:
        sample[ 'noglasses' ] = 1
    return sample
dataset = pd.DataFrame( columns=features )
for subject in subjects:
    # Check that all features exist
    for position in light_positions:
        filename = '.'.join( [subject, position, 'gif'] )
        child_sample = newSample(subject)
        child_sample[ 'filename' ] = filename
        child_sample[ position ] = 1
        dataset = dataset.append( child_sample, ignore_index=True )
    for emotion in emotions:
        filename = '.'.join( [subject, emotion, 'gif'] )
        child_sample = newSample(subject)
        child_sample[ 'filename' ] = filename
        child_sample[ emotion ] = 1
        dataset = dataset.append( child_sample, ignore_index=True )
    for glasses in glasses_present:
        filename = '.'.join( [subject, glasses, 'gif'] )
        child_sample = newSample(subject)
        child_sample[ 'filename' ] = filename
        if glasses == 'noglasses':
            child_sample[ 'glasses' ] = 0
            child_sample[ 'noglasses' ] = 1
        else:
            child_sample[ 'glasses' ] = 1
            child_sample[ 'noglasses' ] = 0
        dataset = dataset.append( child_sample, ignore_index=True )
dataset = dataset.astype( { 'filename':str } )
dataset = dataset.astype( { i:'uint8' for i in subjects } )
dataset = dataset.astype( { i:'uint8' for i in light_positions } )
dataset = dataset.astype( { i:'uint8' for i in emotions } )
dataset = dataset.astype( { i:'uint8' for i in glasses_present } )
print( dataset.dtypes )
dataset.to_csv( 'dataset.csv', index=False )