import torch
# x = torch.rand(5,3)
# print(x)

# print(torch.cuda.is_available())
# print(torch.__version__)


# scalar = torch.tensor(7)
# print(scalar.item())

# rnd_t = torch.randn(size=(3,4,4))
# print(rnd_t.tolist())
# print(rnd_t.shape)
# print(rnd_t.ndim)
# print(rnd_t.dtype)
# print(rnd_t)

# random_img = torch.rand(size=(224,224,3))
# print(random_img.shape,'\n', random_img.ndim)
# zeros = torch.zeros(size=(3,4))
# print(zeros, '\n', zeros.dtype)

data = [1,2,3]
tn = torch.tensor(data, requires_grad=True)
print(tn.shape)