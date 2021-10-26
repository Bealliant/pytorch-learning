from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
'''
        img(PIL format) 
>>>     via method...transforms.Resize(512,512) or transforms.Resize(512) # resize proportionately
            ...Methods Objectified from Class Resize, the function, too, can be a kind of object able to be objectified
-->     img_resized(PIL format)
...     via method...transforms.ToTensor()
            ...Objectified from Class: ToTensor.
            >>> tensor_transform = transforms.ToTensor()
            >>> img_tensor = tensor_transform(img_PIL)

-->     img_tensor(Tensor format)
...     via Tensorboard to print the image       
'''
'''
        use class Compose to shorten the procedure like bellow:
        img(PIL format)

...     activate an instance of transforms.Compose Class
        compose can sum up a series of transforms objects together and operate them orderly.
            >>> trans_compose = transforms.Compose([transforms.Resize(512),transforms.ToTensor()])
            >>> img_tensor_resize = trans_compose(img)
        
-->     output: resized image in Tensor Format
'''

'''
        note::
            One significant thing: we've learnt in OOP(Object Oriented Programming) that we can objectify a class
            and created a precise instance related to the class.
            In this code, we can find out that objects can sometimes be the abstractions of a certain function. The 
            function is first created and later called.
            
'''
# transform.ToTensor
# tensor 的数据类型

img_path = r"dataset/val/bees/2506114833_90a41c5267.jpg"
img_PIL = Image.open(img_path)
tensor_transform = transforms.ToTensor()
# Objectify the tools in the module transform
img_tensor = tensor_transform(img_PIL)
print(img_tensor)
writer = SummaryWriter("logs")
writer.add_image("bees", img_tensor)

trans_normalize = transforms.Normalize([0.5, 0.5, 0.7], [2, 2, 2])
img_norm = trans_normalize(img_tensor)

writer.add_image("new normalized", img_norm)

writer.close()
