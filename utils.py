import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue



import pickle

import os
import time
import torch
import argparse
from tqdm import tqdm
from model import SASRec








    







# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
 

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()




# train/val/test data generation
def data_partitionOLD(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]





def data_partition(fname):
    if fname != 'class_data':
        user_neg = None
        usernum = 0
        itemnum = 0 
        User = defaultdict(list)
        user_train = {}
        user_valid = {}
        user_test = {}
        # assume user/item index starting from 1
        f = open('data/%s.txt' % fname, 'r')
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
    else:
        # load pickle
        import pickle
        with open('data/'+'organic_class_2022_1_3.pkl', "rb") as f:
            data = pickle.load(f)
        class_dat = data['class_dat']
        mat2eng_level = data['mat2eng_level']
        # get max_score for each user. BY group_by user
        user2max_score = dict()
        dat = class_dat.groupby(by='client_sn').max()
        dat['client_sn'] = dat.index
        for element in dat.to_dict('reocrds'):
            client_sn = element['client_sn']
            M_Point = element['M_Point']
            user2max_score[client_sn] = M_Point
        # get user2seq
        user2seq_container = dict()
        for element in class_dat.to_dict('records'):
            client_sn = element['client_sn']
            if client_sn not in user2seq_container:
                user2seq_container[client_sn] = []
            user2seq_container[client_sn].append(element)
        # split train, val, test
        N = 5
        user2train_seq = dict()
        user2val_seq = dict()
        user2test_seq = dict()
        user2val_lv = dict()
        user2test_lv = dict()
        item_set = set()
        user2neg_data = dict()
        # main (remove seq_len less N) + (remove score less max_score)
        for user in list(user2seq_container.keys()):
            lv_data = []
            pos_data = []
            neg_data = []
            seq_container = user2seq_container[user]
            max_score = user2max_score[user]
            for element in seq_container:
                if element['M_Point'] >= max_score:
                    pos_data.append(element['MaterialID'])
                    lv_data.append(element['attend_level'])
                else:
                    neg_data.append(element['MaterialID'])
            if len(pos_data) >= 3:
                train = pos_data[:-2]
                val_test = pos_data[-2:]
                val_test_lv = lv_data[-2:]
                user2train_seq[user] = train
                user2val_seq[user] = [val_test[0]]
                user2test_seq[user] =  [val_test[1]]
                user2val_lv[user] = val_test_lv[0]
                user2test_lv[user] =  val_test_lv[1]
                item_set = item_set | set(pos_data) #| set(neg_data)
                user2neg_data[user] = neg_data
        # usernum
        usernum = len(user2train_seq)
        # itemnum
        itemnum = len(item_set)
        # mapping user, item
        organic_idx2user = dict()
        organic_idx2item = dict()
        ulist = list(user2train_seq.keys())
        user_idx = 1
        item_idx = 1
        for user in ulist:
            organic_idx2user[user] = user_idx
            user_idx +=1
        for item in list(item_set):
            organic_idx2item[item] = item_idx
            item_idx +=1      
        # user_train
        user_train = dict()
        for user in list(user2train_seq.keys()):
            seq = user2train_seq[user]
            user_train[organic_idx2user[user]] = [organic_idx2item[idx] for idx in seq]
        # user_valid
        user_valid = dict()
        for user in list(user2val_seq.keys()):
            seq = user2val_seq[user]
            user_valid[organic_idx2user[user]] = [organic_idx2item[idx] for idx in seq]
        # user_valid
        user_test = dict()
        for user in list(user2test_seq.keys()):
            seq = user2test_seq[user]
            user_test[organic_idx2user[user]] = [organic_idx2item[idx] for idx in seq]
        # user_neg
        user_neg = dict()
        for user in list(user2neg_data.keys()):
            seq = user2neg_data[user]
            user_neg[organic_idx2user[user]] = [organic_idx2item[idx] for idx in seq if idx in organic_idx2item]
    bucket_idx2entity = {'organic_idx2user' : organic_idx2user, 'organic_idx2item' : organic_idx2item}
    return [user_train, user_valid, user_test, usernum, itemnum, bucket_idx2entity]






def explainable_eval(model, u, seq_list, candidate_ys, y, args, dataset):
    [train, valid, test, usernum, itemnum, bucket_idx2entity] = copy.deepcopy(dataset)

    seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1

    for i in reversed(seq_list):
        seq[idx] = i
        idx -= 1
        if idx == -1: break

    pred_element = model.predict(*[np.array(l) for l in [[u], [seq], candidate_ys]], show_attention=True)

    predictions, attention_score = pred_element[0], pred_element[1]

    predictions = predictions[0] # - for 1st argsort DESC

    predictions = list(predictions.tolist())

    item_with_pred = list()
    for i, pred in enumerate(predictions):
        item_with_pred.append([candidate_ys[i], pred])
    item_with_pred = sorted(item_with_pred, reverse=True, key= lambda x:x[1])

    # collect attention score (attention_score : (1, N, N) ; the first N -> time step, the second N -> position)
    
    source_item2attn_score = dict()
    attention_score = attention_score[0][-1]
    for idx in range(len(attention_score[-1])):
        score = attention_score[-1][idx]
        item_id = seq_list[idx]
        source_item2attn_score[item_id] = score
    
    return item_with_pred, source_item2attn_score




def explainable_demo(model, dataset, args):
    # load dataset
    [train, valid, test, usernum, itemnum, bucket_idx2entity] = copy.deepcopy(dataset)

    # random sample one user
    u = random.sample([i+1 for i in range(usernum)], 1)[0]
    
    # build seq_list
    seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1
    seq[idx] = valid[u][0]
    idx -= 1
    for i in reversed(train[u]):
        seq[idx] = i
        idx -= 1
        if idx == -1: break
    seq_list = list(seq)

    # build candidate_ys
    rated = set(train[u])
    rated.add(0)
    candidate_ys = [test[u][0]]
    for _ in range(100):
        t = np.random.randint(1, itemnum + 1)
        while t in rated: t = np.random.randint(1, itemnum + 1)
        candidate_ys.append(t)
    
    # build y
    y = test[u][0]

    # example
    eval_data = {
                'u' : u,
                'seq_list' : seq_list,
                'candidate_ys' : candidate_ys,
                'y' : y,
                'args' : args,
                'dataset' : dataset,
                'model' : model
                }
    # main
    item_with_pred, source_item2attn_score = explainable_eval(**eval_data)

    return item_with_pred, source_item2attn_score, y, u





# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum, bucket_idx2entity] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)


        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()


        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user



# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, bucket_idx2entity] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user






class ArgumentParserSimple:
    def __init__(self, dataset=None, train_dir='default', batch_size=128, 
                       lr=0.001, maxlen=50, hidden_units=50, num_blocks=2,
                       num_epochs=201, num_heads=1, dropout_rate=0.5, l2_emb=0.0, 
                       device='cpu', inference_only=False, state_dict_path=None):
        self.dataset = dataset
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.lr = lr
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_epochs = num_epochs
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.l2_emb = l2_emb
        self.device = device
        self.inference_only = inference_only
        self.state_dict_path = state_dict_path






def load_stanby_model(dataset='class_data', maxlen=200, dropout_rate=0.2, device='cuda', state_dict_path=None):
    args = ArgumentParserSimple(dataset=dataset, maxlen=maxlen, dropout_rate=dropout_rate, device=device, state_dict_path=state_dict_path)

    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum, bucket_idx2entity] = dataset
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    
    
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation? 
     
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)


    model.train() # enable model training


    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    return model, args



