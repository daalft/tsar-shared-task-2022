# -*- coding: utf-8 -*-
"""LS-shared-task-baselines.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11KRCX2jhCU0voFXvw-MRZU-sqfT1zXui
"""

# !pip install transformers

# from transformers import pipeline
# !pip install fasttext
from transformers import pipeline
import transformers

import fasttext
import fasttext.util
import gensim
import glob
import math
import random
import numpy as np
import os
from collections import Counter
# from drive.MyDrive.cental_codes.TSAR2022SharedTaskmain 
# import tsar_eval
import pickle
from tqdm import tqdm

import random

all_ds_tsar = {}
for ds in ["TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_en_trial_gold.tsv",
           "TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_es_trial_gold.tsv",
           "TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_pt_trial_gold.tsv",
           ]:
    with open(ds) as inputfile:
        lang = ds .split("/")[-1].split("_")[1]
        all_ds_tsar[lang] = []
        for ln in inputfile:
            cols = ln.strip().split("\t")
            sent = cols[0]
            target = cols[1]
            reply = Counter(cols[2:])
            assert target in sent
            assert len(sent.split(target)) == 2
            all_ds_tsar[lang].append({"sent": sent, "target": target, "ans": reply, "ds_name": ds})


semeval_ds = {}
for lang, path in {"en": "semeval2012T1/2012.csv", "es": "semeval2012T1/2010_es.csv"}.items():
    semeval_ds[lang] = []
    with open(path) as input_file:
        for ln in input_file:
            if "<head>" not in ln:
                continue
            cols = ln.strip().split("\t")
            # print(cols)
            # input("...")
            sent = cols[0]
            ans = cols[1:]
            if len(sent.split("<head>")) > 3:
                continue
            target = sent.split("<head>")[1]
            # sent = sent.replace("<head>","")
            semeval_ds[lang].append({"sent": sent, "target": target, "ans": ans, "gold": ans, "ds_name": path.split("/")[-1].replace(".csv","")})
            # print(semeval_ds[lang][-1])
            # input("...")
        lst = semeval_ds[lang] 
        lst = lst[int(len(lst)*.8):]
        semeval_ds[lang] = lst
    print(lang,len(semeval_ds[lang]))


####################
####
####    Query expansion
####
####################


# the binary FastText model should by in the TSAR-2022-Shared-Task-main diretory
# complex word is repeated |sent|/2
def __expansion(ds = [("semeval",semeval_ds),("tsar",all_ds_tsar)], buffer_pattern = '_all_langs_fasttextPred_QE_'):
    for ds_name, all_ds in ds:
        if os.path.exists(ds_name + '_all_langs_fasttextPred_QE_'+lang+'.txt'): ## the expansion is quite slow, so we buffer it
            print("loading pregenerated Query Expansion:", ds_name + '_all_langs_fasttextPred_QE_'+lang+'.txt')
            continue
        for lang in all_ds:
            all_langs = {}
            tf_model = gensim.models.fasttext.load_facebook_model('TSAR-2022-Shared-Task-main/cc.'+lang+'.300.bin') # too much memory for preloading
            print('Lang',lang)
            predictions = []
            ds_file = None
            results = {}
            for case in all_ds[lang]:
                lst = tf_model.wv.most_similar(positive=[case["target"]]*int(max(1,round(len(case["sent"].replace("<head>","").split(" "))/2,0))) + case["sent"].replace("<head>","").split(" "), negative=[], topn=5)
                lst = [w for w,s in lst]
                predictions.append({"sent":case["sent"],"target":case["target"], "ans":lst})
                ds_file = case["ds_name"]    
            all_langs[lang] = predictions
            tf_model = None # garbage collector doesn't work properly
            with open(ds_name + buffer_pattern +lang+'.txt', 'w') as output_file:
                for lang in all_langs:
                    for cand in all_langs[lang]:
                        output_file.write(lang+"\t"+cand["sent"]+"\t"+cand["target"]+"\t"+"\t".join(cand["ans"]) + "\n")

 # method: 1 = repeat the entire sent ; 2 = use only the QE words without repeat the entire sent
def __qe_bert_generation(method = 2, buffer_pattern = '_all_langs_fasttextPred_QE_', ds_structure=[("tsar",all_ds_tsar),("semeval",semeval_ds)], bert_models = {"es":"dccuchile/bert-base-spanish-wwm-cased","pt":"neuralmind/bert-base-portuguese-cased","en":"bert-large-uncased",}):
    ds = {}
    # all_langs = {"en":{},"es":{},"pt":{}}
    for file in glob.glob("*" + buffer_pattern+"*"+'.txt'):
        ds_name = file.split("/")[-1].split("_")[0]
        if ds_name not in ds:
            ds[ds_name] = {"en":{},"es":{},"pt":{}}

        with open(file) as input_file:
            for ln in input_file:
                cols = ln.split("\t")
                lang = cols[0]
                sent = cols[1]
                target = cols[2]
                cand = [w.lower() for w in cols[3:] if len(w)>0]
                ds[ds_name][lang][sent + "\t" + target] = cand

    ret = []
    for ds_name, all_ds in ds_structure:
        for lang in all_ds:
            ret.append(ds_name + "_" + lang + "_" + bert_models[lang].replace("/","_") +"_m" +str(method)+ ".txt")
            with open(ds_name + "_" + lang + "_" + bert_models[lang].replace("/","_") +"_m" +str(method)+ ".txt","w") as output_file:
                for multilanguage in [False]:
                    bad_scores = []
                    good_scores = []
                    print("*"*50)
                    print(ds_name, 'Lang', lang, "(M)" if multilanguage else "")
                    print("*"*50)
                    bert = bert_models["m" if multilanguage else lang]
                    bert = pipeline('fill-mask',model=bert, use_fast=True, top_k=20)
                    predictions = []
                    ds_file = None
                    results = {}
                    for case in all_ds[lang]:
                        num_context = 5
                        try_again = True
                        while try_again:
                            try:
                                sequence_to_classify = case["sent"].replace("<head>","") + " " 
                                if method == 1:
                                    sequence_to_classify += " ".join( [ case["sent"].replace("<head>","").replace(case["target"], w) for w in ds[ds_name][lang][case["sent"].replace("<head>","") +"\t" +case["target"] ][:num_context] ] )  
                                elif method == 2:
                                    sequence_to_classify += " ".join( [ w for w in ds[ds_name][lang][case["sent"].replace("<head>","") +"\t" +case["target"] ][:num_context] ] )  
                                if "<head>" in case["sent"]:
                                    sequence_to_classify += " " + case["sent"].replace("<head>"+case["target"]+"<head>", "[MASK]")
                                else:
                                    sequence_to_classify += " " + case["sent"].replace(case["target"], "[MASK]")
                                ans = bert(sequence_to_classify)
                                lst = [x for _,x in sorted([(v["score"],v["token_str"]) for v in ans], reverse = True)]
                                predictions.append({"sent":case["sent"].replace("<head>",""),"target":case["target"], "ans":lst})
                                ds_file = case["ds_name"]
                                output_file.write(case["sent"].replace("<head>","") + "\t" + case["target"] + "\t" + "\t".join(lst) + "\n")
                                try_again = False
                            except:
                                print("skip",ds_name + "_" + lang + "_" + bert_models[lang].replace("/","_") +"_" +str(num_context)+ ".txt")     
                                num_context -= 1
    return ret

def query_expansion(ds = [("semeval",semeval_ds),("tsar",all_ds_tsar)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 2, bert_models = {"es":"dccuchile/bert-base-spanish-wwm-cased","pt":"neuralmind/bert-base-portuguese-cased","en":"bert-large-uncased",}):
    __expansion(ds = ds, buffer_pattern = buffer_pattern)
    files = __qe_bert_generation(method = method, buffer_pattern = buffer_pattern, bert_models = bert_models, ds_structure = ds)
    print("check the following files:")
    for f in files:
        print("\t",f)


####################
####
####    Repeated sentence
####
####################

def double_sentence(use_sep = False, ds = [("semeval",semeval_ds),("tsar",all_ds_tsar)], bert_models = None):

    # bert_models = {"es":[
    #                         "roberta-base", 
    #                         "roberta-large", 
    #                         "skimai/spanberta-base-cased", 
    #                         "PlanTL-GOB-ES/roberta-base-bne",
    #                         "dccuchile/bert-base-spanish-wwm-cased",    #bert
    #                         "dccuchile/bert-base-spanish-wwm-uncased"    #bert
                         
    #                     ],
    #                "pt":[
    #                         "josu/roberta-pt-br",
    #                         "rdenadai/BR_BERTo",
    #                         "neuralmind/bert-base-portuguese-cased",    #bert
    #                         "neuralmind/bert-large-portuguese-cased",    #bert
    #                     ],
    #                "en":[
    #                         "roberta-base",
    #                         "roberta-large",
    #                         "bert-large-uncased",    #bert
    #                         "bert-base-uncased"    #bert
    #                     ],
    #                 }

    for ds_name, all_ds in ds:
        for lang in all_ds:
            print('Lang',lang)
            predictions = {m:[] for m in bert_models[lang]}
            results = {}
            for model_name in bert_models[lang]:
                print(lang,model_name)
                output_file_name = ds_name + "_" + lang + "_" + model_name.replace("/","_") +"_sep" +str(use_sep)+ ".txt"
                with open(output_file_name, "w") as output_file:
                    bert = pipeline('fill-mask',model=model_name, use_fast=True, top_k=20)
                    for case in all_ds[lang]:
                        if use_sep:
                            sequence_to_classify = case["sent"].replace("<head>","").strip() + " [SEP] "+ case["sent"].replace("<head>"+case["target"]+"<head>", "[MASK]" if "bert-" in model_name else "<mask>")
                        else:
                            sequence_to_classify = case["sent"].replace("<head>","").strip() + " "+ case["sent"].replace("<head>"+case["target"]+"<head>", "[MASK]" if "bert-" in model_name else "<mask>")
                        ans = bert(sequence_to_classify)
                        lst = sorted([(v["score"],v["token_str"]) for v in ans], reverse = True)
                        lst = [w.strip() for s, w in lst]
                        output_file.write(case["sent"].replace("<head>","") + "\t" + case["target"]  + "\t" + "\t".join(lst) + "\n")
                        # predictions[model_name].append({"sent":case["sent"],"target":case["target"], "gold":case["gold"], "ans":lst})

bert_models = {"es":[
                        "roberta-base", 
                        "roberta-large", 
                        "skimai/spanberta-base-cased", 
                        "PlanTL-GOB-ES/roberta-base-bne",
                        "dccuchile/bert-base-spanish-wwm-cased",    #bert
                        "dccuchile/bert-base-spanish-wwm-uncased"    #bert
                     
                    ],
               "pt":[
                        "josu/roberta-pt-br",
                        "rdenadai/BR_BERTo",
                        "neuralmind/bert-base-portuguese-cased",    #bert
                        "neuralmind/bert-large-portuguese-cased",    #bert
                    ],
               "en":[
                        "roberta-base",
                        "roberta-large",
                        "bert-large-uncased",    #bert
                        "bert-base-uncased"    #bert
                    ],
                }

double_sentence(use_sep = False, ds = [("semeval",semeval_ds),("tsar",all_ds_tsar)], bert_models = bert_models)

