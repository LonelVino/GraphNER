import numpy as np

from utils.batchify import batchify
from utils.metric import get_ner_fmeasure, integer

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    '''
    Update the learning rate of the optimizer, then return the optimizer
    '''
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def predict_check(pred_variable, gold_variable, mask_variable):
    """
    Args:
        pred_variable (batch_size, sent_len): pred tag result, in numpy format
        gold_variable (batch_size, sent_len): gold result variable
        mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    correct_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return correct_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
    TODO: add comments
    Args:
        pred_variable (batch_size, sent_len): pred tag result
        gold_variable (batch_size, sent_len): gold result variable
        mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idx_2]) for idx_2 in range(seq_len) if mask[idx][idx_2] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idx_2]) for idx_2 in range(seq_len) if mask[idx][idx_2] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label
