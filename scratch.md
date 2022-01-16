# Data Types in Pytorch - Tensor 

In the Pytorch, Data types are little different from Python.

| python       | Pytorch                                                      |
| ------------ | ------------------------------------------------------------ |
| Int          | Int  Tensor of size()                                        |
| float        | Float  Tensor of size()  *e.g.*  *32-bit floating point*  *dtypes: torch.float32*  *CPU tensor torch.Float Tensor* |
| Int  array   | IntTensor  of size [d1,d2,d3,d4]                             |
| Float  array | FloatTensor  of size[d1,d2,…]                                |
| String       | ---                                                          |

 


## dim0: to define scalars

In Pytorch, we will create a scalar by setting a 0-dimension tensor object.

```pycon

a = a.cuda():# to transfer the variable a from CPU to GPU
isinstance(a,torch.cuda.FloatTensor) # to check if the variable is in the GPU.


Out[11]: tensor(2.)

torch.tensor(2.).shape

Out[12]: torch.Size([])

torch.tensor(1.)
Out[13]: tensor(1.)
torch.tensor(1.3000)
Out[14]: tensor(1.3000)

 ```
to use shape/dim/size() to return the magnitude of the variable.

## dim1: to define vectors / tensors

In Pytorch, we call them tensor(张量) , instead of vector (向量)
```pycon
# the statement below we directly assign value to the tensor object.
torch.tensor([1.1])
Out[15]: tensor([1.1000])
torch.tensor([1.1,2.2])
Out[16]: tensor([1.1000, 2.2000])

# or we can use random value to initialize the Tensor Object.
# like the code shown below
a = torch.FloatTensor(4)
Out[18]: tensor([2.7924e-05, 4.5907e-41, 0.0000e+00, 0.0000e+00])

a.type()
Out[19]: 'torch.FloatTensor'

a.size()
Out[20]: torch.Size([4])

print(a)
tensor([1.1000e+00, 3.7911e+22, 6.0028e-02, 4.5907e-41])

import numpy as np
data = np.zeros(2)
data
Out[26]: array([0., 0.])
torch.from_numpy(data)
Out[27]: tensor([0., 0.], dtype=torch.float64)


```

## dim2: Matrix
```pycon
a = torch.tensor([[1,2,3,4],[5,6,7,8]])
# the return form of the size function is a tuple.
a.size(0)
Out[7]: 2
a.size(1)
Out[8]: 4
# the return form of the "shape" method is a list, so use [].
a.shape[1]
Out[9]: 4
```

## dim3: NLP
```pycon
a = torch.rand([1,2,3])
a
Out[15]: 
tensor([[[0.8111, 0.2305, 0.6409],
         [0.3051, 0.2884, 0.1827]]])

# we claim a 1*2*3 tensor object. Given that the first dimension is 1,
# so there is a dual [ in tensor. These can be used in RNN or NLP.

```

## dim4: CNN/CV
the four dimension tensor is very suitable for the Computer Vision.

We used the 4-dimension tensor for CNN, standing for Convolution Neural Network.

[b,c,h,w]:  b: batch;   c: channel: often is RGB 3 channels;    h: height;  w: width;

```pycon
a = torch.rand([2,3,28,28])
a
Out[17]: 
tensor([[[[0.2244, 0.0987, 0.2185,  ..., 0.3864, 0.6830, 0.1338],
          [0.5461, 0.0083, 0.6608,  ..., 0.4380, 0.3668, 0.6682],
          [0.0619, 0.2608, 0.7509,  ..., 0.6381, 0.1938, 0.5845],
          ...,
          [0.7369, 0.8310, 0.0912,  ..., 0.4433, 0.9115, 0.6536],
          [0.2112, 0.4469, 0.3947,  ..., 0.9839, 0.2915, 0.2695],
          [0.3198, 0.3699, 0.9480,  ..., 0.2510, 0.3752, 0.4985]],
         [[0.9395, 0.2641, 0.0044,  ..., 0.9084, 0.2416, 0.7252],
          [0.5532, 0.6948, 0.6968,  ..., 0.9111, 0.5177, 0.8638],
          [0.5250, 0.9055, 0.5223,  ..., 0.0469, 0.7694, 0.4168],
          ...,
          [0.0043, 0.4980, 0.0321,  ..., 0.5542, 0.0407, 0.3214],
          [0.2189, 0.4591, 0.6775,  ..., 0.9006, 0.4542, 0.7491],
          [0.7729, 0.3125, 0.6066,  ..., 0.4674, 0.7814, 0.3467]],
         [[0.0120, 0.1251, 0.6829,  ..., 0.6543, 0.7484, 0.0191],
          [0.8737, 0.1760, 0.7824,  ..., 0.0350, 0.4248, 0.4245],
          [0.2319, 0.1057, 0.4645,  ..., 0.4739, 0.1229, 0.8188],
          ...,
          [0.4162, 0.4090, 0.4329,  ..., 0.9563, 0.6974, 0.1495],
          [0.7277, 0.8996, 0.3618,  ..., 0.5451, 0.1686, 0.0155],
          [0.1575, 0.3094, 0.3888,  ..., 0.0880, 0.4021, 0.0086]]],
        [[[0.0060, 0.0466, 0.3758,  ..., 0.6443, 0.7698, 0.4675],
          [0.7263, 0.7298, 0.2837,  ..., 0.7978, 0.2632, 0.1165],
          [0.7672, 0.8009, 0.8685,  ..., 0.7694, 0.0395, 0.3990],
          ...,
          [0.2351, 0.0115, 0.4971,  ..., 0.8534, 0.2360, 0.3183],
          [0.0919, 0.4280, 0.6880,  ..., 0.6028, 0.1911, 0.4891],
          [0.0067, 0.1894, 0.0250,  ..., 0.7509, 0.1614, 0.6768]],
         [[0.0829, 0.1740, 0.8894,  ..., 0.6371, 0.4575, 0.6270],
          [0.9882, 0.3964, 0.0535,  ..., 0.7510, 0.9562, 0.3002],
          [0.9167, 0.9214, 0.0555,  ..., 0.8082, 0.9696, 0.3929],
          ...,
          [0.2338, 0.2301, 0.4537,  ..., 0.4857, 0.7399, 0.4151],
          [0.9124, 0.1593, 0.3212,  ..., 0.3137, 0.3243, 0.3143],
          [0.7797, 0.6566, 0.5079,  ..., 0.5216, 0.4809, 0.7015]],
         [[0.4569, 0.6406, 0.0478,  ..., 0.6154, 0.3039, 0.8145],
          [0.8986, 0.3216, 0.1551,  ..., 0.6144, 0.9778, 0.2337],
          [0.6696, 0.7516, 0.8898,  ..., 0.3452, 0.6539, 0.3873],
          ...,
          [0.8689, 0.7421, 0.1311,  ..., 0.9170, 0.5029, 0.1259],
          [0.9862, 0.9710, 0.6340,  ..., 0.9150, 0.3485, 0.4342],
          [0.2765, 0.6868, 0.1552,  ..., 0.2679, 0.1852, 0.7606]]]])
a.shape
Out[18]: torch.Size([2, 3, 28, 28])

# return the number of the elements.
a.numel()
Out[19]: 4704
# return the dimension of the tensor.
a.dim()
Out[20]: 4
```

# Establish Your Tensor
## 1. import the Data from Numpy
```pycon
import numpy as np
a = np.array([2,3.3])
torch.from_numpy(a)
Out[23]: tensor([2.0000, 3.3000], dtype=torch.float64)

```
## 2. import the Data from List
<u>**CAUTION**: in torch, **Tensor and tensor have very distinct meanings and usages**.</u>

The Lowered one, tensor, **receives concrete data**, in the "list" form and transform it into the tensor form.

The Capitalized one, Tensor, **receive dimension/ shape without '[]' or '()' ** and initialize by randomizing.

when the parameters are sent in with a **[]** or **()**, it is equivalent to the lowered tensor.

```pycon
torch.FloatTensor([2,3.2])
Out[26]: tensor([2.0000, 3.2000])
torch.FloatTensor((2,3.))   # ATTENTION: double '()' inside!!
Out[27]: tensor([2., 3.]) 

```
## 3. Randomly Initialized Conditions

We will give it a tuple containing its size, and it will randomly output the tensor.

### i. torch.empty() 

```pycon
torch.empty(1)
Out[29]: tensor([1.4013e-45])
torch.Tensor(2,3)
Out[30]: 
tensor([[0.7797, 0.6566, 0.5079],
        [0.5216, 0.4809, 0.7015]])
        
```

### ii. torch.IntTensor/DoubleTensor/FloatTensor/Tensor 

```pycon
torch.IntTensor(2,3)
Out[31]: 
tensor([[1061656548, 1059592065, 1057097732],
        [1057327727, 1056322960, 1060345703]], dtype=torch.int32)

# if we use torch.Tensor(), the output will be determined by the default data type.

# using rand_like(), we will end up with a random array with a similar scale and identical dimension to the input.
b = torch.rand(3,3)
print(b)
tensor([[0.0243, 0.6173, 0.5790],
        [0.1930, 0.0638, 0.6129],
        [0.5597, 0.8656, 0.5074]])
```

### iii. rand_like

using rand_like(), we will end up with a random array with a similar scale and identical dimension to the input.

```pycon

b = torch.rand(3,3)
print(b)
tensor([[0.0243, 0.6173, 0.5790],
        [0.1930, 0.0638, 0.6129],
        [0.5597, 0.8656, 0.5074]])     
a = torch.rand_like(b)

print(a)
tensor([[0.8139, 0.2618, 0.1044],
        [0.1020, 0.6161, 0.6839],
        [0.3427, 0.4591, 0.8161]])



```
There are one thing to pay attention that the torch.rand outputs numbers in the range(0,1)
Also keep in mind that the input of rand_like is a tensor object.

### iv. randint
The formula is like this: 

torch.randint(min,max,[size])

```pycon
torch.randint(1,10,[3,3])
Out[22]: 
tensor([[2, 9, 4],
        [7, 2, 2],
        [2, 4, 7]])
```

### v. randn

To output a normally distributed array, or “满足高斯分布”

The average or the mean of the array is 0, and the Standard Deviation is 1.

## Assignment of Concrete Value

### i. full 

To fill an array with an identical value.
```python
import torch
torch.full([2,3],7) # Function Prototype: torch.full([size],value)
```
Equivalent to `value * ones(m,n)` in MATLAB.
```pycon
torch.full([2,3],7) # to establish a 2*3 dimension matrix with 7.
torch.full([3],7)   # to establish a 3 dimensional vector.
torch.full(7.)      # to establish a scalar. but torch.tensor(7.) is more commonly used.
```

### ii. arange 
!! pay attention to its spelling!!! 
We use torch.arrange() instead of range() to generate a tensor-type increment, but they function similarly.  
In common python codes, we got:
```pycon
range(0,10,2)
Out[24]: range(0, 10, 2)
for i in range(0,10,2):
    print(i)
0
2
4
6
8
```
So the prototype of the function is: range(begin,end,increment)
Keep in mind that the range of function range() is [begin,end).

we got this:
```pycon
torch.arange(0,10,2)
Out[28]: tensor([0, 2, 4, 6, 8])
```
Equivalent in MatLab `(0:2:10)` . The prototype is (begin:increment:end) and the end is not exclusive.

### iii. linspace() / logspace()

to generate a uniformly-spaced list.

Last Update: 30th, Nov, 21

The prototype of the function is: linspace(begin,end,number) to get num points from begin to end.

Pay attention that you should include "steps=" when inputting the 3rd parameter.
```pycon
torch.linspace(0,10,steps=2)
Out[5]: tensor([ 0., 10.])
```

As for logspace, the api will generate numbers whose log values will be uniformly-spaced.

```pycon
torch.logspace(0,1,steps=10)
Out[6]: 
tensor([ 1.0000,  1.2915,  1.6681,  2.1544,  2.7826,  3.5938,  4.6416,  5.9948,
         7.7426, 10.0000])
```
### Ones/Zeros/Eye

Similar to the same function in matlab.

We input the shape and receive a 2-D matrix.
```pycon
a= torch.ones(2,2)
print(a)
tensor([[1., 1.],
        [1., 1.]])
```

### randperm

Similar to the random. shuffle in Numpy module.

Only one **scalar** input and will generate a random list ranging from [0,n-1].

Use the output list as random indexes.
```pycon
torch.randperm(10)
Out[11]: tensor([6, 8, 2, 3, 9, 1, 4, 5, 0, 7])
```

and here is a little example about the idea.

```python
import torch
a = torch.rand(7,2)
b = torch.rand(7,2)
```
and the value of a,b is:
```pycon
a
Out[15]: 
tensor([[0.5723, 0.5954],
        [0.5696, 0.5769],
        [0.1320, 0.3706],
        [0.4746, 0.5989],
        [0.0620, 0.9483],
        [0.0163, 0.7460],
        [0.7576, 0.2381]])
b
Out[16]: 
tensor([[0.9484, 0.7490],
        [0.0396, 0.5273],
        [0.9524, 0.4560],
        [0.4436, 0.8828],
        [0.2635, 0.3716],
        [0.7768, 0.5028],
        [0.7697, 0.4723]])
```
Then we will use randperm() to access both tensor list a and tensor list b.
```python
import torch
a = torch.rand(7,2)
b = torch.rand(7,2)
indx = torch.randperm(7)
ans = a[indx],b[indx]
```
```pycon
ans
Out[18]: 
(tensor([[0.4746, 0.5989],
         [0.0163, 0.7460],
         [0.0620, 0.9483],
         [0.7576, 0.2381],
         [0.5723, 0.5954],
         [0.1320, 0.3706],
         [0.5696, 0.5769]]),
 tensor([[0.4436, 0.8828],
         [0.7768, 0.5028],
         [0.2635, 0.3716],
         [0.7697, 0.4723],
         [0.9484, 0.7490],
         [0.9524, 0.4560],
         [0.0396, 0.5273]]))
indx
Out[19]: tensor([3, 5, 4, 6, 0, 2, 1])
```
As shown, we can get the same position in a and b.