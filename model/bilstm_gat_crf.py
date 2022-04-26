import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from layer.crf import CRF
from layer.gatlayer import GAT
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BLSTM_GAT_CRF(nn.Module):
    def __init__(self, data, args):
        super(BLSTM_GAT_CRF, self).__init__()
        print("build BLSTM_GAT_CRF model...")
        self.name = "BLSTM_GAT_CRF"
        self.strategy = args.strategy
        self.char_emb_dim = data.char_emb_dim
        self.gaz_emb_dim = data.gaz_emb_dim
        self.gaz_embeddings = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.char_embeddings = nn.Embedding(data.char_alphabet.size(), self.char_emb_dim)
        if data.pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
            torch.from_numpy(self.random_embedding(data.char_alphabet.size(), self.char_emb_dim)))
        if data.pretrain_gaz_embedding is not None:
            self.gaz_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embeddings.weight.data.copy_(
            torch.from_numpy(self.random_embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)))
        if args.fix_gaz_emb:
            self.gaz_embeddings.weight.requires_grad = False
        else:
            self.gaz_embeddings.weight.requires_grad = True
        self.hidden_dim = self.gaz_emb_dim
        self.bilstm_flag = args.bilstm_flag
        self.lstm_layer = args.lstm_layer
        if self.bilstm_flag:
            lstm_hidden = self.hidden_dim // 2
        else:
            lstm_hidden = self.hidden_dim
        crf_input_dim = data.label_alphabet.size()+1

        self.lstm = nn.LSTM(self.char_emb_dim, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.char_emb_dim,
                                    out_channels=256,
                                    kernel_size=h),  # 卷积层，这里的Sequential只包含了卷积层，感觉这样写有些多余，
                          # Sequential可以把卷积层、 激活函数层和池化层都包含进去
                          nn.ReLU(),  # 激活函数层
                          nn.MaxPool1d(kernel_size=200 - h + 1))
            # 池化层，这里kernel_size的值应该不用进行这样多余的计算，我们的目的
            # 是将feature maps转为一行一列的向量，那么只要保证这里的kernel_size大于等于feature maps的行数就可以了
            for h in [2,3,4]  # 不同卷积层的kernel_size不一样，注意：这里的kernel_size和池化层的kernel_size是不同的
        ])  # 创建多个卷积层，包含了 图中的convolution、activation function和 maxPooling
        self.fc = nn.Linear(in_features=256 * 3,
                            out_features=300)  # 把最终的特征向量的大小转换成类别的大小，以此作为前向传播的输出。这里和图中的有些区别，貌似并没有用到softmax和regularization

        self.fc1 = nn.Linear(in_features=300, out_features=6)


        self.hidden2hidden = nn.Linear(self.hidden_dim, crf_input_dim)
        self.hidden2hidden1 = nn.Linear(self.hidden_dim, crf_input_dim)
        self.gat_1 = GAT(self.hidden_dim, args.gat_nhidden, crf_input_dim, args.dropgat, args.alpha, args.gat_nhead, args.gat_layer)
        self.gat_2 = GAT(self.hidden_dim, args.gat_nhidden, crf_input_dim, args.dropgat, args.alpha, args.gat_nhead, args.gat_layer)

        self.crf = CRF(data.label_alphabet.size()-1, args.use_gpu)
        if self.strategy == "v":
            self.weight1 = nn.Parameter(torch.ones(crf_input_dim))
            self.weight2 = nn.Parameter(torch.ones(crf_input_dim))
            self.weight3 = nn.Parameter(torch.ones(crf_input_dim))
            self.weight4 = nn.Parameter(torch.ones(crf_input_dim))
        elif self.strategy == "n":
            self.weight1 = nn.Parameter(torch.ones(1))
            self.weight2 = nn.Parameter(torch.ones(1))
            self.weight3 = nn.Parameter(torch.ones(1))
            self.weight4 = nn.Parameter(torch.ones(1))
        else:
            self.weight = nn.Linear(crf_input_dim*4, crf_input_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.droplstm = nn.Dropout(args.droplstm)
        self.gaz_dropout = nn.Dropout(args.gaz_dropout)
        self.reset_parameters()
        if args.use_gpu:
            self.to_cuda()

    def to_cuda(self):
        self.char_embeddings = self.char_embeddings.cuda()
        self.gaz_embeddings = self.gaz_embeddings.cuda()
        self.lstm = self.lstm.cuda()
        self.convs = self.convs.cuda()
        self.fc = self.fc.cuda()
        self.fc1 = self.fc1.cuda()

        self.gat_1 = self.gat_1.cuda()
        self.gat_2 = self.gat_2.cuda()
        self.hidden2hidden = self.hidden2hidden.cuda()
        self.hidden2hidden1 = self.hidden2hidden1.cuda()
        self.gaz_dropout = self.gaz_dropout.cuda()
        self.dropout = self.dropout.cuda()
        self.droplstm = self.droplstm.cuda()
        self.gaz_dropout = self.gaz_dropout.cuda()
        if self.strategy in ["v", "n"]:
            self.weight1.data = self.weight1.data.cuda()
            self.weight2.data = self.weight2.data.cuda()
            self.weight3.data = self.weight3.data.cuda()
            self.weight4.data = self.weight4.data.cuda()
        else:
            self.weight = self.weight.cuda()

    def reset_parameters(self):
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0_reverse)
        nn.init.orthogonal_(self.lstm.weight_ih_l0_reverse)
        nn.init.orthogonal_(self.hidden2hidden.weight)
        nn.init.constant_(self.hidden2hidden.bias, 0)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def _get_lstm_features(self, batch_char, batch_len):
        embeds = self.char_embeddings(batch_char)
        embeds = self.dropout(embeds)
        embeds_pack = pack_padded_sequence(embeds, batch_len, batch_first=True)
        out_packed, (_, _) = self.lstm(embeds_pack)
        lstm_feature, _ = pad_packed_sequence(out_packed, batch_first=True)
        lstm_feature = self.droplstm(lstm_feature)
        return lstm_feature

    def _get_cnn_features(self,batch_char_cnn,batch_len):
        embed_x = self.char_embeddings(batch_char_cnn)  # 将句子转为词向量矩阵，大小为1*7*5，这里的1是表示只有1条句子
        embed_x = self.dropout(embed_x)
        embed_x = embed_x.permute(0, 2, 1)  # 将矩阵转置
        out = [conv(embed_x) for conv in self.convs]  # 计算每层卷积的结果，这里输出的结果已经经过池化层处理了，对应着图中的6 个univariate vectors
        out = torch.cat(out, dim=1)  # 对6 个univariate vectors进行拼接
        out = out.view(-1, out.size(1))  # 按照行优先的顺序排成一个n行1列的数据，这部分在图中没有表现出来，其实在程序运行的过程中6 个univariate vectors的大小可能是1*1*1，进行拼接后是1*1*6，我们在这里将其转换为1*6，可见每个值都没有丢失，只是矩阵的形状变了而已
        #out = F.dropout(input=out,p=self.dropout)  # 这里也没有在图中的表现出来，这里是随机让一部分的神经元失活，避免过拟合。它只会在train的状态下才会生效。进入train状态可查看nn.Module。train()方法

        out = self.fc(out)
        # print('cnn')
        # print(out.size())
        return out

    def _get_attention_features(self,lstm_feature,cnn_feature):
        # print(lstm_feature.size()) #([20, 40, 300])
        # print(cnn_feature.size()) #([20, 300])

        cnn_feature1 = cnn_feature.unsqueeze(-1)
        # print(cnn_feature1.size())
        e=torch.tanh(torch.matmul(lstm_feature,cnn_feature1))
        # print(e.size()) #([20, 40, 1])
        a=torch.nn.functional.softmax(e)
        att_output=torch.mul(lstm_feature,a)
        # print(att_output.size()) #([20, 40, 300])
        return att_output




    # batch_char 和 gaz_list 是一批次的输入，看怎么形成的batch_char，gaz_list保持一致，就不树查找了
    def _get_crf_feature(self, batch_char, batch_char_cnn,batch_len, gaz_list, t_graph, c_graph):
        gaz_feature = self.gaz_embeddings(gaz_list)  #词组的嵌入没有经过网络，直接赋值
        gaz_feature = self.gaz_dropout(gaz_feature)
        lstm_features = self._get_lstm_features(batch_char, batch_len)


        cnn_feature = self._get_cnn_features(batch_char_cnn,batch_len)
        att_feature = self._get_attention_features(lstm_features,cnn_feature)
        att_feature = self.hidden2hidden1(att_feature)


        max_seq_len = lstm_features.size()[1]

        gat_input = torch.cat((lstm_features, gaz_feature), dim=1)

        gat_feature_1 = self.gat_1(gat_input, t_graph)
        gat_feature_1 = gat_feature_1[:, :max_seq_len, :]
        gat_feature_2 = self.gat_2(gat_input, c_graph)
        gat_feature_2 = gat_feature_2[:, :max_seq_len, :]
        lstm_feature = self.hidden2hidden(lstm_features)

        crf_feature = self.weight1 * lstm_feature + self.weight2 * gat_feature_1 + self.weight3 * gat_feature_2 + self.weight4 * att_feature
        return crf_feature

    # def tag_attr(self,feature, label_attr):
    #     # 多任务loss
    #     logits_attr= self.fc1(feature)
    #     # print(logits_attr.size())
    #     # print('logits_attr.size()')
    #     # print(logits_attr.size())  #torch.Size([20, 40, 6])
    #
    #     loss=nn.CrossEntropyLoss()
    #     # print('label_attr')
    #     # print(label_attr.size())
    #
    #     probs = F.softmax(logits_attr, dim=-1)
    #     preds_attr = torch.argmax(probs, dim=-1)
    #     # print('preds_attr.size()')
    #     # print(preds_attr.size())  # torch.Size([20, 40])
    #
    #     #将logits_attr变换顺序  torch.Size([20, 6, 40])
    #
    #     loss_attr = loss(logits_attr.permute(0, 2, 1), label_attr).long()
    #
    #
    #
    #     return preds_attr,loss_attr

#     def neg_log_likelihood(self, batch_char, batch_char_cnn,batch_len, gaz_list, t_graph, c_graph, mask, batch_label,batch_attr,O_index):
#         crf_feature,lstm_features = self._get_crf_feature(batch_char,batch_char_cnn, batch_len, gaz_list, t_graph, c_graph)
#         total_loss = self.crf.neg_log_likelihood_loss(crf_feature, mask, batch_label)
#         #加上attr分类loss
#         _, best_path = self.crf._viterbi_decode(crf_feature, mask)
#         # print('best_path')
#         # print(best_path.size())
# #best_path是不是预测的标签序号,且应该是长度一样的
#         mask_attr=torch.ne(O_index,best_path).long()
#         # print('lstm_feature')
#         # print(lstm_features.size())  #size*batch_seq*dim
#         # mask_attr = torch.FloatTensor(torch.not_equal(best_path, O_index))
#
#         preds_attr,loss_attr=self.tag_attr(lstm_features,batch_attr)  #预测每个字符attr标签序号
#         # print('loss_attr')
#         # print(loss_attr.size())
#         # print(mask_attr.size())
#         # print(mask.size())
#         mask=mask.long()
#         loss1= torch.sum(loss_attr * mask_attr*mask, dim=-1)
#         # print('loss1')
#         # print(loss1.size())
#         loss2=torch.sum(mask_attr*mask, dim=-1) + 1e-5
#         loss_attr=torch.mean(torch.div(loss1, loss2).float())
#         loss_attr=loss_attr.long()
#
#         loss=total_loss+loss_attr
#
#
#
#         return loss,total_loss,loss_attr

    def neg_log_likelihood(self, batch_char, batch_char_cnn,batch_len, gaz_list, t_graph, c_graph, mask, batch_label):
        crf_feature = self._get_crf_feature(batch_char,batch_char_cnn, batch_len, gaz_list, t_graph, c_graph)

        total_loss = self.crf.neg_log_likelihood_loss(crf_feature,mask,batch_label)
        return total_loss




    # def forward(self, batch_char,batch_char_cnn, batch_len, gaz_list, t_graph, c_graph, mask,batch_attr):
    #     crf_feature,lstm_feature = self._get_crf_feature(batch_char, batch_char_cnn,batch_len, gaz_list, t_graph, c_graph)
    #     _, best_path = self.crf._viterbi_decode(crf_feature, mask)
    #
    #     preds_attr,_=self.tag_attr(lstm_feature,batch_attr)
    #     return best_path,preds_attr

    def forward(self, batch_char,batch_char_cnn, batch_len, gaz_list, t_graph, c_graph, mask):
        crf_feature = self._get_crf_feature(batch_char, batch_char_cnn,batch_len, gaz_list, t_graph, c_graph)
        _, best_path = self.crf._viterbi_decode(crf_feature, mask)

        return best_path


