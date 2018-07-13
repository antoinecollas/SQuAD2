from transformer import *
from translator import *
input1 = torch.Tensor([[10,3,2,0,0],[12,14,2,3,0]])
input1 = input1.type(torch.LongTensor)
input2 = torch.Tensor([[1,15,2,0,0],[1,12,8,0,0]])
input2 = input2.type(torch.LongTensor)
# tr = Transformer(100,100,5,5,nb_heads=2)
# print(tr(input1, input2))
tr = Translator(20,20,5,5,nb_layers=2,nb_heads=2,d_model=64,nb_neurons=128)
print(tr.train(input1, input2, nb_epochs=10, batch_size=2))