import numpy as np
import math, torch, time, nltk, sys
import torch.nn as nn
import torch.optim as optim
from transformer.transformer import Transformer
from tensorboardX import SummaryWriter
from tqdm import tqdm

# torch.manual_seed(1)

class Scheduler():
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        #paper
        # lr = d_model**(-0.5) * min(step_num**(-0.5), step_num*(warmup_steps**(-1.5)))
        #tensor2tensor
        lr = 2 * min(1, self.step_num/self.warmup_steps) * (1/math.sqrt(max(self.step_num, self.warmup_steps))) * (1/math.sqrt(self.d_model))
        self.step_num = self.step_num + 1
        return lr

class Translator(nn.Module):
    def __init__(self, vocabulary_size_in, vocabulary_size_out, constants, hyperparams):
        super(Translator, self).__init__()
        self.Transformer = Transformer(vocabulary_size_in, vocabulary_size_out, constants, hyperparams)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.Transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = Scheduler(d_model=hyperparams.D_MODEL, warmup_steps=hyperparams.WARMUP_STEPS)
        self.constants = constants
        self.hyperparams = hyperparams

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit(self, training_steps, data_training, data_eval=None):
        '''
        Arg:
            data_training: iterator which gives two batches: one of source language and one for target language
        '''
        writer = SummaryWriter()
        training_loss, gradient_norm = [], []

        for i in tqdm(range(training_steps)):
            X, Y = next(data_training)
            batch_size = X.shape[0]
            bos = torch.zeros(batch_size, 1).fill_(self.constants.BOS_IDX).to(self.constants.DEVICE, torch.LongTensor)
            translation = torch.cat((bos, Y[:,:-1]),dim=1)
            output = self.Transformer(X, translation)
            output = output.contiguous().view(-1, output.size(-1))
            target = Y.contiguous().view(-1)
            lr = self.scheduler.step()
            for p in self.optimizer.param_groups:
                p['lr'] = lr
            self.optimizer.zero_grad()
            loss = self.criterion(output, target)
            training_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            temp = 0
            for p in self.Transformer.parameters():
                temp += torch.sum(p.grad.data**2)
            temp = np.sqrt(temp.cpu())
            gradient_norm.append(temp)

            if ((i+1)%self.hyperparams.EVAL_EVERY_TIMESTEPS)==0:
                torch.save(self.state_dict(), self.constants.WEIGHTS_FILE)
                writer.add_scalar('0_training_set/loss', np.mean(training_loss), i)
                writer.add_scalar('0_training_set/gradient_norm', np.mean(gradient_norm), i)
                writer.add_scalar('2_other/lr', lr, i)
                training_loss, gradient_norm = [], []

                if data_eval:
                    eval_references = []
                    eval_hypotheses = []
                    for l, (X_batch, Y_batch) in enumerate(data_eval):
                        for i in range(Y_batch.shape[0]):
                            eval_references.append(data_eval.itotok(Y_batch[i]))
                        hypotheses = self.translate(X_batch)
                        for i in range(len(hypotheses)):
                            eval_hypotheses.append(data_eval.itotok(hypotheses[i]))
                    def subwords_to_string(subwords):
                        string = ""
                        for subword in subwords:
                            if subword[-2:] == "@@":
                                string += subword[:-2]
                            elif subword != self.constants.PADDING_WORD:
                                string += subword + " "
                        return string

                    for i, (ref, hyp) in enumerate(zip(eval_references, eval_hypotheses)):
                        eval_references[i] = subwords_to_string(ref)
                        eval_hypotheses[i] = subwords_to_string(hyp)

                    ex_phrases = ''
                    for i, (ref, hyp) in enumerate(zip(eval_references, eval_hypotheses)):
                        ex_phrases = ex_phrases + "\n truth: " + ref + "\n prediction: " + hyp + "\n"
                        if i==4:
                            break

                    BLEU = nltk.translate.bleu_score.corpus_bleu(eval_references, eval_hypotheses)
                    writer.add_scalar('1_eval_set/BLEU', BLEU, i)
                    writer.add_text('examples', ex_phrases, i)

    def translate(self, X):
        '''
        Arg:
            X: batch of phrases to translate: tensor(nb_texts, nb_tokens)
        '''
        self.train(False)
        batch_size, max_seq = X.shape
        max_seq += 10 #TODO: remove hard code
        temp = torch.zeros(batch_size, max_seq).type(torch.LongTensor).to(self.constants.DEVICE)
        temp[:,0] = self.constants.BOS_IDX
        enc = self.Transformer.forward_encoder(X)
        for j in range(1, max_seq):
            output = self.Transformer.forward_decoder(X, enc, temp)
            output = torch.argmax(output, dim=-1)
            temp[:,j] = output[:,j-1]
        #remove padding
        translations = []
        for translation in temp:
            temp2 = []
            for i in range(max_seq):
                if translation[i] == self.constants.PADDING_IDX:
                    break
                if i!=0:
                    temp2.append(translation[i])
            translations.append(temp2)
        return translations