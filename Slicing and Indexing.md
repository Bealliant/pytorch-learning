# Indexing and Slicing
## with Scalar
we regard a as a CIFAR image. 
batch size : 4, channel ; channels 3  (RGB); 28,28 width and height 
```pycon
a= torch.rand(4,3,28,28)
a[0].shape
Out[21]: torch.Size([3, 28, 28])
a[0,0].shape
Out[22]: torch.Size([28, 28])
a[0,0,2,4]
Out[23]: tensor(0.3965)
```
## normal slicing way 
Similar to MATLAB, ':' also means from ... to ...
We call it "Unspecified Indexing", to be more precise, it should be "group indexing".

Some rules are shown below.


*******
for one dimension of a matrix
**[a:b] from a to b; b excluded.**

**[:b]  from the first to b; b excluded.**

**[a:]  from a to end.**

**[:]   select all.**

**[-1:] from the last one to the end.** 

**[0:28:2] from 0 to 28 step 2**

**[::2] start from the first one to the end, step 2**   

******

## with .index_select()
.index_select method to Index with a tensor type matrix.
ATTENTION: expected input type of `* (int dim, Tensor index)` instead of `(int, list)`

```pycon
import torch
a= torch.rand(4,3,28,28)
a.index_select(0,torch.tensor([0,2]))
# this means that in the 0 dimension, we choose the element whose index is within the list [0,1].
# torch.arange can be used as well here.
a.index_select(2,torch.arange(8))
```

## slicing with "..."
here "..." means there will be multiple dimensions in which all elements will be chosen.
but the number of dimensions is determined by the position of this denoting.
****

`a[...]`  **equals** `a[:] `        |        **select all of them**

`a[0,...]` **equals**`a[0]`         | **select a[0] dimension**

`a[:,1,...]`    **equals `a[:,1,:,:]`**   |   **select a[all][1][all][all]**
****

As a matter of fact, this denoting "..." will help in conditions where ":," will repeat to appear,
particularly in multi-dimensional matrices.

