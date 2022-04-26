#encoding=utf-8
import jieba
from pyltp import Postagger, Parser
import sys
import codecs

# Setting the default output encoding
sys.stdout = codecs.getwriter('GBK')(sys.stdout.detach()) 
# Since Python 3.7 the encoding of standard streams can be changed with reconfigure():
# sys.stdout.reconfigure(encoding='GBK')


# golden_list = ['O','O','B-GPE.NAM','I-GPE.NAM','I-GPE.NAM','O','O','O','O','B-GPE.NAM','I-GPE.NAM','O','O']  # 一句话的标签
# golden_list_attr = ['NULL','NULL','J','y','J','NULL','NULL','NULL','NULL','G','y','NULL','NULL']
#
# list=[]
# for idy in range(len(golden_list)):
#
#     if golden_list[idy][0] =="B" or golden_list[idy][0] =="I":
#         list.append(golden_list[idy] + "-" + golden_list_attr[idy])
#         # k=idy
#         # while(idy!=len(golden_list)-1):
#         #     if golden_list[idy+1][0]!="I":
#         #         break
#         #     else:
#         #         list.append(golden_list[idy+1] + "-" + golden_list_attr[k])
#         #     idy+=1
#
#
#     else:
#         list.append("O")
# print(list)
# list_new=[]
# for k in range(len(list)):
#     if list[k][0]=='I' and k!=0 and list[k-1]!="O":
#         list_new.append(list[k].split('-')[0]+'-'+list[k].split('-')[1]+'-'+list_new[k-1].split('-')[-1])
#     else:
#         list_new.append(list[k])
#
# print(list_new)

gazs = list(jieba.cut("老师要是提供资料就好了"))     # 使用精确模式对文本进行分词

# Load the Postagger model and Parser model
pos_model_path = 'model/ltp_data_v3.4.0/pos.model'
postagger = Postagger(pos_model_path)   # part-of-speech tagging model
par_model_path = 'model/ltp_data_v3.4.0/parser.model'
parser = Parser(par_model_path)  # Dependency Parsing

# Use Postagger model and Parser Model 
postags = postagger.postag(gazs)   # part-of-speech tagging
arcs = parser.parse(gazs, postags)
rely_id = [arc.head for arc in arcs] # 列表，每个词对应的词的id，从1开始，0表示root
relation = [arc.relation for arc in arcs]
heads = ['Root' if id == 0 else gazs[id-1] for id in rely_id]

for i in range(len(gazs)):
    print(relation[i] + '(' + gazs[i] + ', ' + heads[i] + ')')

