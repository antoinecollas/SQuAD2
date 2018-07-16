from layers import *
from transformer import *
from translator import *

if False:
    print("====TEST MultiHeadAttention====")
    model = MultiHeadAttention(h=2,d_model=5)
    print(list(model.parameters())) #on obtient les paramètres W_q, W_k, W_v, W_o et les deux listes de paramètres de layer_norm
    Q=K=V = torch.Tensor([[[1,2,3,4,5],[1,3,2,0,0]],[[1,2,3,4,5],[12,14,2,3,0]]])
    output = model(Q,K,V)
    output[0][0][0].backward()
    print("W_k grad=", model.W_k.grad)
    print("W_o grad=",model.W_o.grad)

if False:
    print("====TEST Embedding====")
    model = Embedding(vocabulary_size=10, d_model=5)
    print(list(model.parameters()))
    X = torch.tensor([[1,2,3],[1,3,2]])
    output = model(X)
    output[0][0][0].backward()
    print(model.lookup_table.weight.grad)

if False:
    print("====TEST Encoder====")
    model = Encoder(nb_layers=2, nb_heads=2, d_model=5, nb_neurons = 10, dropout=0.1)
    # print(list(model.parameters()))
    X = torch.Tensor([[[1,2,3,4,5],[1,3,2,3,2],[0,0,0,0,0]],[[1,2,3,4,5],[12,14,2,3,0],[0,0,0,0,0]]])
    output = model(X)
    output[0][0][0].backward()
    print(model.MultiHeadAttention[0].W_k.grad)
    print(model.MultiHeadAttention[0].W_o.grad)
    print(model.PositionWiseFeedForward[0].W_1.grad)

if False:
    print("====TEST Embedding + Encoder + Mask====")
    X = torch.Tensor([[1,2,3,4,2],[1,3,2,0,0]]).type(torch.LongTensor)
    model1 = Embedding(vocabulary_size=10, d_model=5)
    model2 = Encoder(nb_layers=2, nb_heads=2, d_model=5, nb_neurons = 10, dropout=0.1)
    mask = get_mask(X,X)
    # print(mask)
    output = model1(X)
    output = model2(output,mask)
    output[0][0][0].backward()

if False:
    print("====TEST Transformer====")
    model = Transformer(vocabulary_size_in=10, vocabulary_size_out=10, nb_tokens_in=5, nb_tokens_out=5, nb_layers=2, nb_heads=2, d_model=12, nb_neurons = 24, dropout=0.1)
    X = torch.Tensor([[1,2,3,4,2],[1,3,2,0,0]]).type(torch.LongTensor)
    target = torch.Tensor([[1,0,0,0,0],[2,0,0,0,0]]).type(torch.LongTensor)
    output = model(X,target)
    output[0][0].backward()

if True:
    print("====TEST Translator====")
    input1 = torch.Tensor([[10,3,2,1,4],[12,14,2,6,7]])
    input1 = input1.type(torch.LongTensor)
    input2 = torch.Tensor([[1,15,2,9,4],[1,12,8,11,6]])
    input2 = input2.type(torch.LongTensor)
    # tr = Transformer(100,100,5,5,nb_heads=2)
    # print(tr(input1, input2))
    tr = Translator(20,20,5,5,nb_layers=1,nb_heads=2,d_model=32,nb_neurons=64)
    tr.train(input1, input2, nb_epochs=100, batch_size=2)
    print(tr.predict(input1))