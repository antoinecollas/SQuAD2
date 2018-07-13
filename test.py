from transformer import *
input1 = torch.Tensor([[10,3,2,0,0],[22,52,2,3,0]])
input1 = input1.type(torch.LongTensor)
input2 = torch.Tensor([[1,15,2,0,0],[1,52,8,0,0]])
input2 = input2.type(torch.LongTensor)
tr = Transformer(100,100,5,5,nb_heads=2)
print(tr.forward(input1, input2))