import numpy as np


def seq_gaz(batch_gaz_ids):

    gaz_len = []
    gaz_list = []
    for gaz_id in batch_gaz_ids:
        gaz = []
        length = 0
        for ele in gaz_id:
            if ele:
                length = length + len(ele[0])
                for j in range(len(ele[0])):
                    gaz.append(ele[0][j])
        gaz_list.append(gaz)
        gaz_len.append(length)
    # print('end')
    # print(gaz_list)
    # print(gaz_len)
    return gaz_list, gaz_len, max(gaz_len)


def graph_generator_new(input):

    max_gaz_len, max_seq_len, gaz_ids_new, gaz_ids_ori, rely_id = input  #单位是一句话

    # print('view')
    # print(gaz_ids_ori)  #gaz_ids_ori=[[16, 2], [1649, 2], [1650,2],。。。]
    # print(gaz_ids_new)  #gaz_ids_new应该=[16,1649,....]
    # print(max_gaz_len) #75
    # print(max_seq_len) #143


    # gaz_seq = []
    gaz_len = len(gaz_ids_new)
    sentence_len=0
    for ele in gaz_ids_ori:
        sentence_len += ele[1]
    # sentence_len = len(gaz_ids)
    # gaz_len = 0
    # for ele in gaz_ids:
    #     if ele:
    #         gaz_len += len(ele[0]) #gaz_len 代表一句话中匹配到了多少个词组
    # print(gaz_len)  #14

    matrix_size = max_gaz_len + max_seq_len
    t_matrix = np.eye(matrix_size, dtype=int)
    # l_matrix = np.eye(matrix_size, dtype=int)
    c_matrix = np.eye(matrix_size, dtype=int)

    add_matrix1 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix2 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix1[:sentence_len, :sentence_len] = np.eye(sentence_len, k=1, dtype=int)
    # print('ma')
    # print(add_matrix1)
    add_matrix2[:sentence_len, :sentence_len] = np.eye(sentence_len, k=-1, dtype=int)
    t_matrix = t_matrix + add_matrix1 + add_matrix2  #字符之间有连接
    # l_matrix = l_matrix + add_matrix1 + add_matrix2
    # give word a index

    # word_id = [[]] * sentence_len
    # word_id = [[]] * gaz_len
    index_gaz = max_seq_len
    index_char=0
    accu=0

    for i in range(gaz_len):
        ele = gaz_ids_ori[i] #[id,len]
        accu+=ele[1]
        #词和上一个字符，下一个字符之间
        if accu<sentence_len:
            t_matrix[index_gaz, accu] = 1
            t_matrix[accu, index_gaz] = 1
        if accu-1-ele[1] >= 0:
            t_matrix[index_gaz, accu-1-ele[1]] = 1
            t_matrix[accu-1-ele[1], index_gaz] = 1
        #依存句法词之间
        if rely_id[i]-1>=0:
            t_matrix[max_seq_len+rely_id[i] - 1, index_gaz] = 1
            t_matrix[index_gaz, max_seq_len + rely_id[i] - 1] = 1
        #词和包含的字符之间
        for k in range(ele[1]):
            c_matrix[index_gaz, index_char] = 1
            c_matrix[index_char, index_gaz] = 1
            index_char+=1
        index_gaz+=1


 #word_id 只涉及到词与词之间的连接
    # for i in range(sentence_len):
    #     if gaz_ids_ori[i]: #gaz_ids=[[], [[496], [2]], [], [[1636], [3]], [[1637], [2]], [], [[1638, 1639], [3, 2]],...]
    #         word_id[i] = [0] * len(gaz_ids_ori[i][1])
    #         for j in range(len(gaz_ids_ori[i][1])):
    #             word_id[i][j] = index
    #             index = index + 1
    # print('result')
    # print(word_id)  #[[143], [], [144], [145], [], [146], [147], [148], [], [], [149], [150], [151], [], [], [152], [], [153, 154], [], [155], [], [], [], [], [156], []]


    # index_gaz = max_seq_len  #词组的序号
    # index_char = 0   #字符的序号
    # 自己建立矩阵 ，现在有的是 #gaz_ids_ori应该=[[[16], [2]], [[1649], [2]], [[1650], [2]], [[38], [2]], [[1651], [2]],。。。]
    # gaz_ids_new应该=[16,1649,....]
    #1.C graph ,将词组中包含的字符都与词组相连


    # for k in range(len(gaz_ids)):
    #     ele = gaz_ids[k]
    #     if ele:
    #         for i in range(len(ele[0])):
    #             gaz_seq.append(ele[0][i])  #gaz_seq 里面是一句话匹配到的所有词组
    #             l_matrix[index_gaz, index_char] = 1
    #             l_matrix[index_char, index_gaz] = 1
    #             l_matrix[index_gaz, index_char + ele[1][i] - 1] = 1
    #             l_matrix[index_char + ele[1][i] - 1, index_gaz] = 1
    #             for m in range(ele[1][i]):
    #                 c_matrix[index_gaz, index_char + m] = 1
    #                 c_matrix[index_char + m, index_gaz] = 1
    #             # char and word connection
    #             if index_char > 0:
    #                 t_matrix[index_gaz, index_char - 1] = 1
    #                 t_matrix[index_char - 1, index_gaz] = 1
    #
    #                 if index_char + ele[1][i] < sentence_len:
    #                     t_matrix[index_gaz, index_char + ele[1][i]] = 1
    #                     t_matrix[index_char + ele[1][i], index_gaz] = 1
    #             else:
    #                 t_matrix[index_gaz, index_char + ele[1][i]] = 1
    #                 t_matrix[index_char + ele[1][i], index_gaz] = 1
    #             # word and word connection
    #             #词和词相邻改成依存句法关系上的相对距离
    #             if index_char + ele[1][i] < sentence_len:
    #                 if gaz_ids[index_char + ele[1][i]]:
    #                     for p in range(len(gaz_ids[index_char + ele[1][i]][1])):
    #                         q = word_id[index_char + ele[1][i]][p]
    #                         t_matrix[index_gaz, q] = 1
    #                         t_matrix[q, index_gaz] = 1
    #             index_gaz = index_gaz + 1
    #     index_char = index_char + 1
    return t_matrix, c_matrix
