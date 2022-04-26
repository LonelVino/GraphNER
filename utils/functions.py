import numpy as np
from pyltp import Postagger, Parser
import sys
import codecs

# Setting the default output encoding
#TODO: Disable in Notebook currently
# sys.stdout = codecs.getwriter('GBK')(sys.stdout.detach()) 
# Since Python 3.7 the encoding of standard streams can be changed with reconfigure():
# sys.stdout.reconfigure(encoding='GBK')


def normalize_word(word):
    '''
    Normalize digital characters as char '0', keep others
    '''
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, gaz, char_alphabet, label_alphabet, gaz_alphabet, number_normalized, max_sent_length):
    in_lines = open(input_file, 'r',encoding='utf-8').readlines()
    instance_texts = []
    instance_ids = []
    chars = []
    labels = []
    char_ids = []
    label_ids = []
    cut_num = 0
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            char = pairs[0]
            if number_normalized:
                char = normalize_word(char)
            label = pairs[-1]
            chars.append(char)
            labels.append(label)
            char_ids.append(char_alphabet.get_index(char))
            label_ids.append(label_alphabet.get_index(label))
        else:
            if ((max_sent_length < 0) or (len(chars) < max_sent_length)) and (len(chars) > 0):
                gazs = []
                gaz_ids = []
                s_length = len(chars)
                for idx in range(s_length):
                    matched_list = gaz.enumerateMatchList(chars[idx:])
                    matched_length = [len(a) for a in matched_list]
                    gazs.append(matched_list)  #一句话匹配到的词列表
                    # print('gazs')
                    # print(gazs)
                    # print(k)
                    matched_id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_id:
                        gaz_ids.append([matched_id, matched_length])
                    else:
                        gaz_ids.append([])  #gaz_ids 和 gazs 的格式都是[[[1210], [2]], [[1211,1341], [2,3]], [], [[1212], [2]], ...]形式的
                instance_texts.append([chars, gazs, labels]) #char列表，词列表，标签
                instance_ids.append([char_ids, gaz_ids, label_ids])
            elif len(chars) < max_sent_length:
                cut_num += 1
            chars = []
            labels = []
            char_ids = []
            label_ids = []
            gazs = []
            gaz_ids = []
    return instance_texts, instance_ids, cut_num

def read_instance_new(input_char_file, input_word_file, input_label_file, char_alphabet, label_alphabet, gaz_alphabet, number_normalized, max_sent_length):
    pos_model_path = 'model/ltp_data_v3.4.0/pos.model'
    postagger = Postagger(pos_model_path)

    par_model_path = 'model/ltp_data_v3.4.0/parser.model'
    parser = Parser(par_model_path)

    char_lines = open(input_char_file, 'r',encoding='utf-8').readlines()
    word_lines = open(input_word_file, 'r', encoding='utf-8').readlines()
    label_lines = open(input_label_file, 'r', encoding='utf-8').readlines()
    # attr_lines=open(input_attr_file,'r',encoding='utf-8').readlines()

    instance_texts = []
    instance_ids = []
    
    for idx in range(len(char_lines)):
        chars = []
        labels = []
        char_ids = []
        label_ids = []
        # attr_ids=[]
        # o_ids=[]
        gazs = []
        gaz_ids = []
        gazs_ori = []
        gaz_ids_ori = []

        char_line = char_lines[idx] #老师上课认真
        word_line = word_lines[idx] #老师 上课 认真
        #对每句话进行分析，形成[[id,id],[id,id]]

        label_line = label_lines[idx]
        # attr_line=attr_lines[idx]
        if number_normalized:
            chars = [normalize_word(char) for char in char_line.strip().split('\t')]
        else:
            chars = char_line.strip().split('\t')
        gazs = word_line.strip().split('\t') #[老师，上课，认真]
        # print('gazs')
        # print(gazs)

        postags = postagger.postag(gazs)
        arcs = parser.parse(gazs, postags)
        rely_id = [arc.head for arc in arcs] #列表，每个词对应的词的id，从1开始，0表示root

        labels = label_line.strip().split('\t')
        # attrs=attr_line.strip().split('\t')

        for char in chars:
            char_ids.append(char_alphabet.get_index(char))
        # for attr in attrs:
        #     attr_ids.append(attr_alphabet.get_index(attr))
        for label in labels:
            label_ids.append(label_alphabet.get_index(label))
            # o_ids.append(label_alphabet.get_index("O"))
        for gaz in gazs:
            # gaz_ids.append([gaz_alphabet.get_index(gaz),len(gaz)])
            gaz_ids.append(gaz_alphabet.get_index(gaz)) #[1，5，8]
            # print(gaz_ids)
            # gazs_ori.append([list(gaz),list(len(gaz))])
            gazs_ori.append([gaz, len(gaz)])

            # gaz_ids_ori.append([list(gaz_alphabet.get_index(gaz)),list(len(gaz))])
            gaz_ids_ori.append([gaz_alphabet.get_index(gaz), len(gaz)])
            # print(gaz_ids_ori)

#这里为了之后形成矩阵，可以设置两种形式的gaz，一种[[1,2],[156,2]，一种跟char保持一致

        instance_texts.append([chars, gazs,gazs_ori, labels])  # char列表，词列表，标签
        instance_ids.append([char_ids, gaz_ids,gaz_ids_ori,rely_id, label_ids])

        # if len(chars) < max_sent_length:
        #     cut_num += 1

    return instance_texts, instance_ids

def read_instance_multi(input_char_file, input_word_file, input_label_file,input_attr_file, char_alphabet, label_alphabet,attr_alphabet, gaz_alphabet, number_normalized, max_sent_length):
    pos_model_path = 'model/ltp_data_v3.4.0/pos.model'
    postagger = Postagger(pos_model_path)
    # postagger.load(pos_model_path)

    par_model_path = 'model/ltp_data_v3.4.0/parser.model'
    parser = Parser(par_model_path)
    # parser.load(par_model_path)

    char_lines = open(input_char_file, 'r',encoding='utf-8').readlines()
    word_lines = open(input_word_file, 'r', encoding='utf-8').readlines()
    label_lines = open(input_label_file, 'r', encoding='utf-8').readlines()
    attr_lines=open(input_attr_file,'r',encoding='utf-8').readlines()

    instance_texts = []
    instance_ids = []

    cut_num = 0

    for idx in range(len(char_lines)):
        chars = []
        labels = []
        char_ids = []
        label_ids = []
        attr_ids=[]
        o_ids=[]
        gazs = []
        gaz_ids = []
        gazs_ori = []
        gaz_ids_ori = []

        char_line = char_lines[idx] #老师上课认真
        word_line = word_lines[idx] #老师 上课 认真
        #对每句话进行分析，形成[[id,id],[id,id]]

        label_line = label_lines[idx]
        attr_line=attr_lines[idx]
        if number_normalized:
            chars = [normalize_word(char) for char in char_line.strip().split('\t')]
        else:
            chars = char_line.strip().split('\t')
        gazs = word_line.strip().split('\t') #[老师，上课，认真]
        # print('gazs')
        # print(gazs)

        postags = postagger.postag(gazs)
        arcs = parser.parse(gazs, postags)
        rely_id = [arc.head for arc in arcs] #列表，每个词对应的词的id，从1开始，0表示root

        labels = label_line.strip().split('\t')
        attrs=attr_line.strip().split('\t')

        for char in chars:
            char_ids.append(char_alphabet.get_index(char))
        for attr in attrs:
            attr_ids.append(attr_alphabet.get_index(attr))
        for label in labels:
            label_ids.append(label_alphabet.get_index(label))
            o_ids.append(label_alphabet.get_index("O"))
        for gaz in gazs:
            # gaz_ids.append([gaz_alphabet.get_index(gaz),len(gaz)])
            gaz_ids.append(gaz_alphabet.get_index(gaz)) #[1，5，8]
            # print(gaz_ids)
            # gazs_ori.append([list(gaz),list(len(gaz))])
            gazs_ori.append([gaz, len(gaz)])

            # gaz_ids_ori.append([list(gaz_alphabet.get_index(gaz)),list(len(gaz))])
            gaz_ids_ori.append([gaz_alphabet.get_index(gaz), len(gaz)])
            # print(gaz_ids_ori)

#这里为了之后形成矩阵，可以设置两种形式的gaz，一种[[1,2],[156,2]，一种跟char保持一致

        instance_texts.append([chars, gazs,gazs_ori, labels,attrs])  # char列表，词列表，标签
        instance_ids.append([char_ids, gaz_ids,gaz_ids_ori,rely_id, label_ids,attr_ids,o_ids])

        # if len(chars) < max_sent_length:
        #     cut_num += 1

    return instance_texts, instance_ids

def build_pretrain_embedding(embedding_path, alphabet, skip_first_row=False, separator=" ", embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path, skip_first_row, separator)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for alph, index in alphabet.iteritems():
        if alph in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph])
            else:
                pretrain_emb[index, :] = embedd_dict[alph]
            perfect_match += 1
        elif alph.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[alph.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding: %s\n     pretrain num:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    embedding_path, pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path, skip_first_row=False, separator=" "):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r',encoding='utf-8') as file:
        i = 0
        j = 0
        for line in file:
            if i == 0:
                i = i + 1
                if skip_first_row:
                    _ = line.strip()
                    continue
            j = j+1
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(separator)
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 == len(tokens):
                    embedd = np.empty([1, embedd_dim])
                    embedd[:] = tokens[1:]
                    embedd_dict[tokens[0]] = embedd
                else:
                    continue
    return embedd_dict, embedd_dim
