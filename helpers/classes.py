from .preprocess_data import total_valid_postiveIDs
import matplotlib.pylab as plt
import numpy as np
import os

class Assay():
    def __init__(self, ID, data_dir):
        self.ID = ID
        self.imagefiles = self.get_imagefiles()
        self.label = 0
        self.data_dir = data_dir
        self.processed_images = None
        self.random_samples = None
        self.label = self.ID in total_valid_postiveIDs
        self.images = None
        self.image_sections = None
        
    def get_imagefiles(self):
        prefixes = ['{}_{}'.format(self.ID,x) for x in ['s1','s2','s3','s4']]
        image_ids = [(y + '_w1.tif', y + '_w2.tif') for y in prefixes]
        return image_ids
    
    def read_rawimages(self):
        images = []
        for rep in self.imagefiles:
            im = np.zeros([2160,2160,3])
            R = plt.imread(os.path.join(self.data_dir, rep[0])) / 65535.
            G = plt.imread(os.path.join(self.data_dir, rep[1])) / 65535.
            B = np.zeros_like(R)
            im[:,:,0], im[:,:,1], im[:,:,2] = R, G, B
            images.append(im)
        images = np.array(images).astype(np.float16)
        self.images = images
        return images
    
    def display(self):
        f, a = plt.subplots(1,4, figsize=(15,4))
        if self.images is None:
            self.images = self.read_rawimages()
        f.suptitle("Drug ID: {}, {}".format(self.ID, ['Negative','Positive'][int(self.label)]), size=20)
        for i in range(4):
            a[i].imshow(self.images[i])
            a[i].axis('off')
            
    def display_sections(self,n):
        f, a = plt.subplots(1,4, figsize=(15,5))
        
        if self.images is None:
            self.images = self.read_rawimages()
        self.image_sections = self.create_sections(n)
        f.suptitle("ID: {}, carried through: {}".format(self.ID, self.label), size=20)
        for i in range(4):
            a[i].imshow(self.images[i][1080-(n/2):1080+(n/2),1080-(n/2):1080+(n/2)])
            a[i].axis('off')
            
    def create_sections(self,n):
        if self.images is None:
            self.images = self.read_rawimages()
        return np.array([image[1080-(n/2):1080+(n/2),1080-(n/2):1080+(n/2)] for image in self.images])
            

