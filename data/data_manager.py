
import os, cv2
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from data.data_sets import FolderDataset
from utils.util import list_dir, load_image, load_audio





class FolderDataManager(object):

    def __init__(self, config):

        load_formats = {
                'image':load_image,
                'audio':load_audio
                }

        assert np.sum(list(config['splits'].values())) >= .999, "Splits must add up to 1"
        assert config['format'] in load_formats, "Pass valid data format"

        self.dir_path = config['path']
        self.loader_params = config['loader']

        self.splits = config['splits']

        self.load_func = load_formats[config['format']]
                
        data_dic, self.mappings, self.classes = self._get_dic()
        self.class_counts = self._class_counts(data_dic)
        
        path_splits = os.path.join(self.dir_path, '.splits.json')
        if os.path.isfile(path_splits):
            self.data_splits = torch.load(path_splits)
        else:         
            data_arr = self._get_arr(data_dic)        
            self.data_splits = self._get_splits(data_arr)
            torch.save(self.data_splits, path_splits) 


    def _get_splits(self, arr):
        np.random.seed(0)
        ret = {s:[] for s in self.splits.keys()}
        split_vec = np.concatenate([[s]*round(len(arr)*p) for s,p in self.splits.items()])
        np.random.shuffle(split_vec)

        for i in range(len(arr)):
            ret[split_vec[i]].append( arr[i] )

        return ret

    def _get_dic(self):

        ret = {}

        classes = list_dir(self.dir_path)

        class_to_idx = dict(zip(classes, np.arange(len(classes))))
        idx_to_class = dict(zip(np.arange(len(classes)), classes))
        mappings = {'idx_to_class':idx_to_class, 'class_to_idx':class_to_idx}

        for c in classes:
            c_path = os.path.join(self.dir_path, c)
            ret[c] = []

            for n in list_dir(c_path):
                ret[c].append( os.path.join(c_path, n) )
 
        return ret, mappings, classes
    

    def _get_arr(self, data_dic):
        ret = [];
        for c, paths in data_dic.items():
            for path in paths:
                class_idx = self.mappings['class_to_idx'][c]
                ret.append({'path':path, 'class':c, 'class_idx':class_idx})
        return ret

    def _class_counts(self, data_dic):
        ret = {}
        for k,v in data_dic.items():
            ret[k] = len(v)
        return ret


    def get_loader(self, name, transfs):
        assert name in self.data_splits
        dataset = FolderDataset(self.data_splits[name], load_func=self.load_func, transforms=transfs)

        return data.DataLoader(dataset=dataset, **self.loader_params)

if __name__ == '__main__':

    pass




