from transformer import *
input1 = torch.Tensor([[1,3,2,0,0],[1,52,2,0,0]])
input1 = input1.type(torch.LongTensor)
tr = Transformer(100,5,5,nb_heads=2)
print(tr.forward(input1))