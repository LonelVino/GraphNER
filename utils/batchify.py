#coding=utf-8
import torch
from utils.graph_generator import *


def batchify(input_batch_list, gpu):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    # print('word')
    # print(words)

    gazs_list = [sent[1] for sent in input_batch_list]
    gaz_lens = [len(g) for g in gazs_list]
    max_gaz_len = max(gaz_lens)

    gazs_ori = [sent[2] for sent in input_batch_list]  #[[id,len],[if,len],...]

    relys_id = [sent[3] for sent in input_batch_list]

    labels = [sent[4] for sent in input_batch_list]

    # attrs=[sent[5] for sent in input_batch_list]

    # o_ids=[sent[6] for sent in input_batch_list]

    word_seq_lengths = list(map(len, words))
    max_seq_len = max(word_seq_lengths)
    max_seq_len1=200



    # print(gazs_list)
    # gazs_list, gaz_lens, max_gaz_len = seq_gaz(gazs)
    # print(k)
    #
    # tmp_matrix=[]
    # for k in range(len(gazs_list)):
    #     gaz_list=gazs_list[k]
    #     gaz_ori=gazs_ori[k]
    #     rely_id=relys_id[k]
    #     # tmp_matrix.append(list(map(graph_generator_new,[(max_gaz_len, max_seq_len, gaz_list, gaz_ori,rely_id)])))
    tmp_matrix = list(map(graph_generator_new, [(max_gaz_len, max_seq_len, gaz_list,gaz_ori,rely_id) for gaz_list, gaz_ori,rely_id in zip(gazs_list,gazs_ori,relys_id)]))

    # print('size')
    # print(np.array(tmp_matrix).shape)
    batch_t_matrix = torch.ByteTensor([ele[0] for ele in tmp_matrix])
    batch_c_matrix = torch.ByteTensor([ele[1] for ele in tmp_matrix])
    # batch_l_matrix = torch.ByteTensor([ele[2] for ele in tmp_matrix])

    gazs_tensor = torch.zeros((batch_size, max_gaz_len), requires_grad=False).long()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    word_seq_tensor_cnn = torch.zeros((batch_size, max_seq_len1), requires_grad=False).long()
    # 全是O的矩阵
    # O_tag_index=torch.zeros((batch_size, max_seq_len), requires_grad=False).long()


    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    # attr_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()


    mask = torch.zeros((batch_size, max_seq_len), requires_grad=False).byte()
    # for idx, (seq, gaz, gaz_len, label,attr, seqlen,o_ids_index) in enumerate(zip(words, gazs_list, gaz_lens, labels,attrs, word_seq_lengths,o_ids)):
    for idx, (seq, gaz, gaz_len, label, seqlen) in enumerate(zip(words, gazs_list, gaz_lens, labels, word_seq_lengths)):
        # O_tag_index[idx, :seqlen]=torch.LongTensor(o_ids_index)
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        word_seq_tensor_cnn[idx, :seqlen] = torch.LongTensor(seq)
        gazs_tensor[idx, :gaz_len] = torch.LongTensor(gaz)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        # attr_seq_tensor[idx, :seqlen] = torch.LongTensor(attr)

        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths = torch.LongTensor(word_seq_lengths)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    word_seq_tensor_cnn = word_seq_tensor_cnn[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]

    # attr_seq_tensor = attr_seq_tensor[word_perm_idx]
    # O_tag_index=O_tag_index[word_perm_idx]

    gazs_tensor = gazs_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    batch_t_matrix = batch_t_matrix[word_perm_idx]
    batch_c_matrix = batch_c_matrix[word_perm_idx]
    # batch_l_matrix = batch_l_matrix[word_perm_idx]
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_tensor_cnn = word_seq_tensor_cnn.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()

        # attr_seq_tensor = attr_seq_tensor.cuda()
        #
        # O_tag_index=O_tag_index.cuda()
        mask = mask.cuda()
        batch_t_matrix = batch_t_matrix.cuda()
        gazs_tensor = gazs_tensor.cuda()
        batch_c_matrix = batch_c_matrix.cuda()
        # batch_l_matrix = batch_l_matrix.cuda()
    # return word_seq_tensor, word_seq_tensor_cnn, word_seq_lengths, gazs_tensor, mask, label_seq_tensor,attr_seq_tensor, word_seq_recover, batch_t_matrix, batch_c_matrix,O_tag_index
    return word_seq_tensor, word_seq_tensor_cnn, word_seq_lengths, gazs_tensor, mask, label_seq_tensor, word_seq_recover, batch_t_matrix, batch_c_matrix


