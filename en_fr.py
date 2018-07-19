from translator import *
import sys
sys.path.append("../fastai/")
from fastai.text import *
from constants import *

'''
    Translation from English to French
'''

ENGLISH_FILENAME = "training-giga-fren/giga-fren.release2.fixed_100.en"
FRENCH_FILENAME = "training-giga-fren/giga-fren.release2.fixed_100.fr"

def prepare_texts(filename, lang):
    with open(filename,"r") as f:
        phrases = f.readlines()
    tok = Tokenizer(lang=lang).proc_all_mp(partition_by_cores(phrases))
    freq = Counter(p for o in tok for p in o)
    # print("Most common words:", freq.most_common(25))
    itos = [o for o,c in freq.most_common(MAX_VOCAB) if c>MIN_FREQ]
    itos.insert(0, EOS_WORD)
    itos.insert(0, BOS_WORD)
    itos.insert(0, UNKNOW_WORD)
    itos.insert(0, PADDING_WORD)
    # print("Length dictionnary integer to string=", len(itos))
    #we use a default value when the string doesn't exist in the dictionnary
    stoi = collections.defaultdict(lambda:UNKNOW_WORD_IDX, {v:k for k,v in enumerate(itos)})
    texts = np.array([[stoi[o] for o in p] for p in tok])
    return texts, itos, stoi

def pad_batch(batch, length=None):
    if length is None:
        len_max = -float('Inf')
        for phrase in batch:
            if len(phrase)>len_max:
                len_max = len(phrase)
    else:
        len_max=length
    result = np.zeros((batch.shape[0], len_max))
    k=0
    for phrase in batch:
        for i in range(len(phrase), len_max):
            phrase.append(PADDING_IDX)
        result[k] = np.array(phrase)
        k+=1
    return result

texts_en, itos_en, stoi_en = prepare_texts(ENGLISH_FILENAME, lang='en')
texts_fr, itos_fr, stoi_fr = prepare_texts(FRENCH_FILENAME, lang='fr')

batches_idx = list(SortishSampler(texts_en, key=lambda x: len(texts_en[x]), bs=BATCH_SIZE))

nb_texts = len(texts_en)
nb_batches = nb_texts//BATCH_SIZE

tr = Translator(vocabulary_size_in=len(stoi_en),vocabulary_size_out=len(stoi_fr),max_seq=MAX_SEQ,nb_layers=NB_LAYERS,nb_heads=NB_HEADS,d_model=D_MODEL,nb_neurons=NB_NEURONS)
tr.train(True)
tr.to(DEVICE)

for k in range(NB_EPOCH):
    print("=======Epoch:=======",k)
    for l in range(nb_batches):
        print("Batch:",l)
        X_batch = torch.from_numpy(pad_batch(texts_en[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]])).type(torch.LongTensor).to(DEVICE)
        Y_batch = torch.from_numpy(pad_batch(texts_fr[batches_idx[l*BATCH_SIZE:(l+1)*BATCH_SIZE]], length=MAX_SEQ)).type(torch.LongTensor).to(DEVICE)
        tr.fit(X_batch, Y_batch)

