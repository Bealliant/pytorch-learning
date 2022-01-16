# Dimension Shift
## 1. view / shape
view and shape are basically identical

Provided that the whole size of the tensor object has not been changed.

"view" is available as long as `prod(a.size) = prod(a'.size)`
```pycon
a= torch.rand(4,1,28,28)
a.view(4,28*28)
Out[7]: 
tensor([[0.7760, 0.3059, 0.2804,  ..., 0.7751, 0.7342, 0.4924],
        [0.8976, 0.0152, 0.4747,  ..., 0.3701, 0.5952, 0.0335],
        [0.8061, 0.3951, 0.8979,  ..., 0.9738, 0.8502, 0.3749],
        [0.7535, 0.7787, 0.6563,  ..., 0.1859, 0.4392, 0.2409]])
```

But we should pay attention that the operation 'view' will damage some dimensional info.

Some dimensional info may be lost.

## 2. squeeze / unsqueeze
### unsqueeze(idx = None)

The range of the Position Index can be [`-a.dim-1(),a.dim()`]
```pycon
a.shape
Out[8]: torch.Size([4, 1, 28, 28])
a.unsqueeze(0).shape
Out[9]: torch.Size([1, 4, 1, 28, 28])
a.unsqueeze(-1).shape
Out[10]: torch.Size([4, 1, 28, 28, 1])
```
**positive Positional index : ahead**

Here is another example:
```pycon
b = torch.rand(32)
f= torch.rand(4,32,14,14)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
b.size()
Out[15]: torch.Size([1, 32, 1, 1])
```
So after the expansion, we can add tensor b and tensor f together.

### squeeze(idx = None)

If the number of the dimension is equal to one, then this dimension is able to squeeze.

If not, the matrix will remain the same.

The idx is the index of dimension upon which we want to squeeze.

And None will perform squeeze to all dimensions.

A few examples:
```pycon
a.squeeze(1).size()
Out[16]: torch.Size([4, 28, 28])
a.size()
Out[17]: torch.Size([4, 1, 28, 28])
b.squeeze().size()
Out[20]: torch.Size([32])
b.size()
Out[21]: torch.Size([1, 32, 1, 1])
```

## 3. transpose .t() / permute 
### .t()
available only to 2D tensor.
```pycon
b.size()
Out[30]: torch.Size([4, 3])
b.t().shape
Out[31]: torch.Size([3, 4])
```

**For 3D or 4D matrices, we use transpose(dim0,dim1)**

```pycon
a.shape
Out[32]: torch.Size([4, 1, 28, 28])
a.transpose(0,1).shape
Out[34]: torch.Size([1, 4, 28, 28])
```

### permute
If there is a batch of images in [b,c,h,w] form, how to reform them into [b,h,w,c] ?

1. Use the transpose(dim0,dim1) we learned previously
```pycon
a = torch.rand(4,3,28,32)
a.transpose(1,3).transpose(1,2).shape
Out[36]: torch.Size([4, 28, 32, 3])
```
2. use permute(*dims), similar to list indexing

***dims is the index of the previous dimensions**
```pycon
a.permute(0,2,3,1).shape
Out[37]: torch.Size([4, 28, 32, 3])
```
## 4. expand / repeat(*args)
**Expand : broadcasting**

**Repeat : memories copied**

Therefore, Expand has more advantages on the aspect of CPU memory saving.

### expand
Input the expected output shape.

```pycon
b.shape
Out[22]: torch.Size([1, 32, 1, 1])
b.expand(-1,32,28,28).size()
Out[24]: torch.Size([1, 32, 28, 28])
```
Here "-1" means remain the original dimension number.
*args is the expected DN(dim num) after processing.

### repeat

Input the times of repetition.

```pycon
b.shape
Out[25]: torch.Size([1, 32, 1, 1])
b.repeat(1,1,28,28).shape
Out[28]: torch.Size([1, 32, 28, 28])
```

The expected output DN is b.shape .* args

# BroadCasting

auto-expansion with data uninvolved.

Difference in size can be ignored when operating.

Is it broadcasting-able ?

A[......]
B[...]
If current dim = 1, expand to the same.

If either has no dim, insert one dim and expand to the same.

Otherwise, Not broadcasting-albe.

