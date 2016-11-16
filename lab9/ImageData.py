from PIL import Image
import numpy as np


class ImageData( object ):

    def __init__(self, inFile):
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

        # shuffle the data
        permutation = np.random.permutation(len(labels))
        data = data[permutation]
        labels = labels[permutation]
        cut_off = int(len(labels) * .72)
        self.train_data = data[:cut_off,:]
        self.train_labels = labels[:cut_off,:]
        self.test_data = data[cut_off:,:]
        self.test_labels = labels[cut_off:,:]
        self.batch_count = 0


    def _shuffle(self):
        permutation = np.random.permutation(len(self.train_labels))
        self.train_data = self.train_data[permutation]
        self.train_labels = self.train_labels[permutation]

    def getBatch(self, batch_size=128):
        start = batch_size * self.batch_count
        self.batch_count += 1
        end = batch_size * self.batch_count 
        if len(self.train_labels[start:end]) < batch_size:
            self._shuffle()
            self.batch_count = 1
            return self.train_data[0:batch_size], self.train_labels[0:batch_size]
        return self.train_data[start:end], self.train_labels[start:end]

