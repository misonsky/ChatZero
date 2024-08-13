#coding=utf-8
from typing import List
import json
import argparse
from collections import OrderedDict
from tqdm import tqdm
import os
from utils.EvaluationUtils import calculationEmbedding,cal_Distinct,compute_bleu_rouge_single_prediction,cal_corpus_bleu

parser = argparse.ArgumentParser('parameters config for evaluation')
parameters_settings = parser.add_argument_group('parameters settings')
parameters_settings.add_argument('--embedding',type=str,default="glove",help="golve word2 vector")
parameters_settings.add_argument('--corpus',type=str,default="DailyDialog",help='select task to train')
parameters_settings.add_argument('--embedding_size',type=int,default=300,help="GoogleNews embeddings for evaluation")
parameters_settings.add_argument('--result_file',type=str,default="predictionsResults/dstc/svt_dstc.json",help="the result filenames")
config=parser.parse_args()

def EvaluateDialogue(config,golds:List[str],predictions:List[str]):
    golds = [gold.split() for gold in golds]
    predictions = [prediction.split() for prediction in predictions]
    metricsResults = calculationEmbedding(config,predictions=predictions,references=golds)
    metricsResults.update(cal_corpus_bleu(golds,predictions))
    metricsResults.update(cal_Distinct(predictions))
    golds_dict = { _index:[" ".join(gold)] for _index, gold in enumerate(golds)}
    con_dict = { _index:[" ".join(pred)] for _index, pred in enumerate(predictions)}
    metricsResults.update(compute_bleu_rouge_single_prediction(con_dict,golds_dict))
    return metricsResults

def get_predictions():
    total_num = 0
    predictions,targets = [],[]
    with open(config.result_file,"r",encoding="utf-8") as f:
        results= json.load(f)
    for item in results:
        token_pred = item["pred"].split()#word_tokenize(item["pred"])
        total_num += len(token_pred)
        predictions.append(" ".join(token_pred))
        targets.append(" ".join(item["target"].split()))#
    print(total_num*1.0/len(predictions))
    return predictions,targets

def evaluate_metrics():
    results=[]
    # emb_dic = OrderedDict()
    # embedding = os.path.join(config.embedding,config.corpus,"en","vectors.txt")
    # with open(embedding,"r",encoding="utf-8") as f:
    #     for line in f:
    #         line  = line.rstrip()
    #         line = line.split()
    #         token = str(line[0])
    #         vector = [float(item) for item in line[-config.embedding_size:]]
    #         assert len(vector) == config.embedding_size
    #         emb_dic[token] = vector
    predictions,ground_response = get_predictions()
    metricsResult = EvaluateDialogue(config,ground_response,predictions)
    formatString="the prediction metrics is "
    for _key in metricsResult:
        formatString +=_key
        formatString +=" {} ".format(metricsResult[_key])
    print(formatString)
evaluate_metrics()
    



