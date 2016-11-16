from PIL import Image
import numpy as np


class ImageData
def load_data(inFile):
    '''
    assumes list.txt is a list of filenames, formatted as
       ./lfw2//Aaron_Eckhart/Aaron_Eckhart_0001.jpg
       ./lfw2//Aaron_Guiel/Aaron_Guiel_0001.jpg
       ...

    Returns:
      data (13233, 250, 250)
      labels (13233, 1)
     '''
    files = open( inFile ).readlines()
    
    data = np.zeros(( len(files), 250, 250 ))
    labels = np.zeros(( len(files), 1 ))
    
    # a little hash map mapping subjects to IDs
    ids = {}
    scnt = 0
    
    # load in all of our images
    ind = 0
    for fn in files:
    
        subject = fn.split('/')[3]
        if not ids.has_key( subject ):
            ids[ subject ] = scnt
            scnt += 1
        label = ids[ subject ]
    
        data[ ind, :, : ] = np.array( Image.open( fn.rstrip() ) )
        labels[ ind ] = label
        ind += 1

    return data, labels

data, labels = load_data('./list.txt')
permutation = np.random.permutation(len(data))
data = data[permutation]
labels = labels[permutation]
ratio = .75
split = int(len(data) * ratio)
train_data, train_lables = data[:split], labels[:split]
test_data, test_labels = data[split:], labels[split:]
print len(train_lables)
