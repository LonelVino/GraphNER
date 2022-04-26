#coding=utf-8
from utils.data import Data
from utils.config import get_args
from model.bilstm_gat_crf import BLSTM_GAT_CRF

import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import time
import random
import sys
import gc
import warnings
warnings.filterwarnings("ignore")


def data_initialization(args):
    data_stored_directory = args.data_stored_directory
    file = data_stored_directory + args.dataset_name + "_new_dataset.dsetCOMBINE"
    if os.path.exists(file) and not args.refresh:
        print('\n[INFO]: Existed Dataset Detected.')
        data = load_data_setting(data_stored_directory, args.dataset_name)
    else:
        print('\n[INFO]: No existed dataset, start generating a new dataset.')
        data = Data()
        data.dataset_name = args.dataset_name
        data.norm_char_emb = args.norm_char_emb
        data.norm_gaz_emb = args.norm_gaz_emb
        data.number_normalized = args.number_normalized
        data.max_sentence_length = args.max_sentence_length
        # data.build_gaz_file(args.gaz_file) # sgns.merge.word 是词组和对应300维向量 得到self.gaz，不确定格式
        # data.generate_instance_new(args.train_char_file, args.train_word_file, args.train_label_file,args.train_attr_file, "train", False)
        # data.generate_instance_new(args.dev_char_file, args.dev_word_file, args.dev_label_file,args.test_attr_file, "dev")
        # data.generate_instance_new(args.test_char_file, args.test_word_file, args.test_label_file,args.test_attr_file, "test")

        # data.generate_instance_new(args.train_char_file, args.train_word_file, args.train_label_file, "train", False)
        # data.generate_instance_new(args.dev_char_file, args.dev_word_file, args.dev_label_file, "dev")
        # data.generate_instance_new(args.test_char_file, args.test_word_file, args.test_label_file, "test")

        data.generate_instance_new(args.train_char_file, args.train_word_file, args.train_file, "train", False)
        data.generate_instance_new(args.dev_char_file, args.dev_word_file, args.dev_file, "dev")
        data.generate_instance_new(args.test_char_file, args.test_word_file, args.test_file, "test")


        data.build_char_pretrain_emb(args.char_embedding_path) #返回self.pretrain_char_embedding, self.char_emb_dim
        data.build_gaz_pretrain_emb(args.gaz_file)
        data.fix_alphabet()
        data.get_tag_scheme()
        save_data_setting(data, data_stored_directory)
    return data


def save_data_setting(data, data_stored_directory):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(data_stored_directory):
        os.makedirs(data_stored_directory)
    dataset_saved_name = data_stored_directory + data.dataset_name +"_new_dataset.dsetCOMBINE"
    with open(dataset_saved_name, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", dataset_saved_name)


def load_data_setting(data_stored_directory, name):
    dataset_saved_name = data_stored_directory + name + "_new_dataset.dsetCOMBINE"
    with open(dataset_saved_name, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", dataset_saved_name)
    data.show_data_summary()
    return data







def evaluate(data, model, args, name):
    if name == "train":
        instances = data.train_ids
    elif name == "dev":
        instances = data.dev_ids
    elif name == 'test':
        instances = data.test_ids
    else:
        print("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []

    # pred_results_attr = []
    # gold_results_attr = []

    model.eval()
    batch_size = args.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        # char,char_cnn, c_len, gazs, mask, label, attr,recover, t_graph, c_graph,o_index= batchify(instance, args.use_gpu)
        # tag_seq,tag_attr = model(char,char_cnn, c_len, gazs, t_graph, c_graph, mask,attr)

        char, char_cnn, c_len, gazs, mask, label, recover, t_graph, c_graph = batchify(instance,args.use_gpu)
        tag_seq= model(char, char_cnn, c_len, gazs, t_graph, c_graph, mask)

        pred_label, gold_label = recover_label(tag_seq, label, mask, data.label_alphabet, recover)
        #pred_label:标签序列[O,B-,O,I-,]
        pred_results += pred_label
        gold_results += gold_label

        #attr标签转换
        # pred_attr, gold_attr = recover_label(tag_attr, attr, mask, data.attr_alphabet, recover)
        # # pred_label:标签序列[null,教学方法,。。。,]
        # pred_results_attr += pred_attr
        # gold_results_attr += gold_attr

        #将两种标签序列合并成一个 O就不结合了，B-和I-开头的选取对应的attr类别多的赋值【O,O,B-实体-教学态度，I-实体-教学态度，】




    decode_time = time.time() - start_time
    speed = len(instances)/decode_time

    # pred = integer(pred_results, pred_results_attr)
    # gold = integer(gold_results, gold_results_attr)
    # acc = get_acc(pred, gold)

    # acc, p, r, f = get_ner_fmeasure(gold, pred, data.tagscheme)
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagscheme)

    return speed, acc, p, r, f,pred_results


def train(data, model, args):
    '''
    Execute the training process on model
    '''
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    best_dev = -1
    for idx in range(args.max_epoch):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, args.max_epoch))
        optimizer = lr_decay(optimizer, idx, args.lr_decay, args.lr)
        instance_count = 0
        sample_loss = 0
        total_loss = 0
        # sample_bio_loss=0
        # sample_attr_loss = 0
        random.shuffle(data.train_ids)
        model.train()
        model.zero_grad()
        batch_size = args.batch_size
        train_num = len(data.train_ids)
        # print('train_num')
        # print(train_num)
        total_batch = train_num // batch_size + 1
        # print('total_batch')
        # print(total_batch)
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > train_num:
                end = train_num
            instance = data.train_ids[start:end]
            # print('instance')
            if not instance:
                continue
            model.zero_grad()
            # char, char_cnn,c_len, gazs, mask, label,attr, recover, t_graph, c_graph ,o_index = batchify(instance, args.use_gpu)
            # #
            # loss,bio_loss,attr_loss = model.neg_log_likelihood(char, char_cnn, c_len, gazs, t_graph, c_graph, mask, label,attr,o_index)
            # tag_seq,tag_attr= model(char, char_cnn, c_len, gazs, t_graph, c_graph, mask)
            # pred_label, _ = recover_label(tag_seq, label, mask, data.label_alphabet, recover)
            # model能输出crf和lstm特征并给neg_log吗
            # 不输出特征，会经过两次模型，第一次输出预测结果给了第二次
            char, char_cnn, c_len, gazs, mask, label, recover, t_graph, c_graph = batchify(instance,args.use_gpu)
            loss = model.neg_log_likelihood(char, char_cnn, c_len, gazs, t_graph, c_graph, mask, label)


            instance_count += 1
            # sample_bio_loss+=bio_loss.item()
            # sample_attr_loss += attr_loss.item()
            sample_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            if args.use_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            model.zero_grad()
            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f" % (end, temp_cost, sample_loss))
                sys.stdout.flush()
                sample_loss = 0
                # sample_bio_loss=0
                # sample_attr_loss=0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f" % (end, temp_cost, sample_loss))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        speed, acc, p, r, f,_ = evaluate(data, model, args, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        current_score = f
        print(
            "Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
        if current_score > best_dev:
            print("Exceed previous best f score:", best_dev)
            if not os.path.exists(args.param_stored_directory + args.dataset_name + "_param"):
                os.makedirs(args.param_stored_directory + args.dataset_name + "_param")
            model_name = "{}epoch_{}_f1_{}.model".format(args.param_stored_directory + args.dataset_name + "_param/", idx, current_score)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
        gc.collect()


if __name__ == '__main__':
    args, unparsed = get_args()
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    # Set the enviroment of Torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    seed = args.random_seed
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # Import data
    data = data_initialization(args)
    # Initialize and Train Model
    model = BLSTM_GAT_CRF(data, args)
    train(data, model, args)


