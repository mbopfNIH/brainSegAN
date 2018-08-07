from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import SimpleITK as sitk
import math
from glob import glob

''' labels:
    1 for necrosis
    2 for edema
    3 for non-enhancing tumor
    4 for enhancing tumor
    0 for everything else
'''

# DIR = 'sample/' #sample dir
# D_path = 'data/'
# HGG_path = 'data/HGG/'
# LGG_path = 'data/LGG/'
# threshold = 1280#int(mean + 3*stddev)
# ratio = threshold/255

def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)

    if title:
        plt.title(title)

    plt.show()

class Preprocess(object):
    def __init__(self, path, n4itk = True, normalize = True, cnt = 0):
        self.path = '/A/Brats/data/train/' + path
        self.n4itk = n4itk
        self.normalize = normalize
        self.cnt = cnt

    def N4BFC(self, img):
        img = sitk.Cast(img, sitk.sitkFloat32)
        img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0))   ## Create a mask spanning the part containing the brain, as we want to apply the filter to the brain image
        corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
        return corrected_img

    def read_data(self):
        foldernames = [f for f in sorted(listdir(self.path)) if not isfile(join(self.path, f))]
        cnt = self.cnt

        for folder in foldernames:
            path = self.path + '/' + folder + '/'
            filenames = [f for f in sorted(listdir(path)) if not isfile(join(path, f))]
            # images = []
            # labels = []
            img_temp_t1 = np.array([])
            img_temp_t1c = np.array([])
            img_temp_flair = np.array([])
            img_temp_t2 = np.array([])
            label_temp = np.array([])
            for key in filenames:
                if 'OT' in key:
                    img = sitk.ReadImage(path + key + '/' + key + '.mha')
                    nda = sitk.GetArrayFromImage(img)
                    label_temp = np.float32(nda)
                    # for i in range(nda.shape[0]):
                    #     labels.append(nda[i,:,:])
                else:
                    img = sitk.ReadImage(path + key + '/' + key + '.mha')
                    if self.n4itk:
                        img = self.N4BFC(img)
                    nda = sitk.GetArrayFromImage(img)
                    if self.normalize:
                        nda = self._normalize(nda)
                    if 'T1c' in key:
                        img_temp_t1c = np.float32(nda)
                    if 'T1' in key and 'T1c' not in key:
                        img_temp_t1 = np.float32(nda)
                    if 'Flair' in key:
                        img_temp_flair = np.float32(nda)
                    if 'T2' in key:
                        img_temp_t2 = np.float32(nda)

            idx = key.split('.')[5]
            for i in range(nda.shape[0]):
                if i > 9 and i < 145:
                    a = img_temp_t1c[i,:,:]
                    b = img_temp_t1[i,:,:]
                    c = img_temp_flair[i,:,:]
                    d = img_temp_t2[i,:,:]
                    e = label_temp[i,:,:]
                    img_temp = np.stack([a,b,c,d,e],axis=0)
                    # print(img_temp.shape)
                    self.save_data(img_temp, idx, i)
                    # images.append(img_temp)

            # images = np.array(images)
            cnt = cnt + 1
            if cnt % 10 == 0:
                print('Load %d groups' % cnt)
        # return images, labels

    def _normalize(self, volume):
        '''
        INPUT:  (1) a single slice of any given modality (excluding gt)
                (2) index of modality assoc with slice (0=t1c, 1=t1, 2=flair, 3=t2)
        OUTPUT: normalized slice
        '''
        t = np.percentile(volume, 99.5)
        volume = np.clip(volume, 0, t)
        if np.std(volume) == 0:
            volume =  volume
        else:
            volume =  (volume - np.mean(volume)) / np.std(volume)
        volume = (volume - np.max(volume))/-np.ptp(volume)
        volume = 1 - volume
        # print(np.max(volume))
        return volume



    def save_data(self, images, name, idx):

        outfile = './train/' + name + '_' + str(idx) + '.npy'
        # with open(outfile, 'wb') as f_out:
        #     pickle.dump(images, f_out, protocol=2)
        np.save(outfile, images)
            # print('save to: ', outfile)

        # outfile = outpath + 'labels.pickle'
        # with open(outfile, 'wb') as f_out:
        #     pickle.dump(labels, f_out)
        #     print('save to: ', outfile1)


if __name__ == '__main__':
    # out_dir = os.path.join(data_path, 'train/')
    out_dir = './train'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    images = []
    images_temp = []
    labels = []
    labels_temp = []

    print('Loading LGG......')
    LGG = Preprocess('LGG', n4itk = False)
    LGG.read_data()
    # images_temp, labels_temp = LGG.read_data()
    # images.append(images_temp)
    # labels.append(labels_temp)
    # for key in foldernames:
    #     path = inpath + key + '/'
    #     count += 1
    #     max_value = read_data(path, images, labels, count)
    #     if max_value >= vmax:
    #         vmax = max_value
    print('Loading HGG......')
    HGG = Preprocess('HGG', n4itk = False)
    HGG.read_data()
    # images_temp, labels_temp = HGG.read_data()
    # images.append(images_temp)
    # labels.append(labels_temp)
    # for key in foldernames2:
    #     path = HGG_path + key + '/'
    #     count += 1
    #     max_value = read_data(path, images, labels, count)
    #     if max_value >= vmax:
    #         vmax = max_value
    # print('Saving......')

    # save_data(images, labels, out_dir)
