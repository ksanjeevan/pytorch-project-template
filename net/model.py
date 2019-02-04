import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

# F.max_pool2d needs kernel_size and stride. If only one argument is passed, 
# then kernel_size = stride

class ToyCNN(BaseModel):
    def __init__(self, classes):
        super(ToyCNN, self).__init__()
        self.flatten_size = 20*53*53
        self.classes = classes

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.flatten_size, 50)
        self.fc2 = nn.Linear(50, len(classes))


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # This 20*53*53 is a function of the input size. I think
        # this should be handled by the data_manager and fed into the model 
        # on initialization.
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        ret = F.log_softmax(x, dim=1)

        return ret 


import requests
import torchvision.models as models
class VGG16(BaseModel):
    def __init__(self, classes=None, mode='', state_dict=None):

        super(VGG16, self).__init__(mode)
        self.classes = classes
        self.model = models.vgg16(pretrained=True)

        tl_mode = self.mode
        if tl_mode == 'init':
            self._modify_last_layer(len(self.classes))
        elif tl_mode == 'freeze':
            self._tl_freeze()
            self._modify_last_layer(len(self.classes))
        elif tl_mode == 'random':
            self._tl_random()
            self._modify_last_layer(len(self.classes))
        elif state_dict is not None:
            out_size, _ = self._get_shape_from_sd(state_dict)
            self._modify_last_layer(out_size)
        else:
            self.classes = self.__get_labels()


    def __get_labels(self):
        LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
        index_to_class = {int(key):value for (key, value)
           in requests.get(LABELS_URL).json().items()}
        return [index_to_class[k] for k in sorted(list(index_to_class.keys()))]


    def _modify_last_layer(self, num_classes):
        in_dim = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_dim, num_classes)
    

    def _tl_freeze(self):

        for param in self.model.parameters():
            param.requires_grad = False

    def _tl_random(self):
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def _get_shape_from_sd(self, dic):
        '''
        Get last layer shape when loading from a state dict.
        '''
        return dic['model.classifier.6.weight'].shape

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        out_raw = self.forward( x.unsqueeze(0) )
        out = F.softmax(out_raw, dim=1)
        max_ind = out.argmax().item()

        return self.classes[max_ind], out[:,max_ind].item()


