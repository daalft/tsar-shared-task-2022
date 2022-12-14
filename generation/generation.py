#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""LS-shared-task-baselines.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/11KRCX2jhCU0voFXvw-MRZU-sqfT1zXui
"""

# !pip install transformers

# from transformers import pipeline
#!pip install fasttext
#!pip install gensim
from transformers import pipeline
import transformers
import torch
from transformers import BertTokenizer, RobertaTokenizer, BertModel, BertForMaskedLM, PegasusForConditionalGeneration, PegasusTokenizer, AutoModel, AutoModelWithLMHead, AutoTokenizer
import logging
import re
import fasttext
import fasttext.util
import gensim
import glob
import math
import random
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
# from drive.MyDrive.cental_codes.TSAR2022SharedTaskmain 
# import tsar_eval
import pickle
logging.basicConfig(level=logging.INFO)# OPTIONAL

all_ds_tsar_final = {}
for lang, ds in [("en", "../corpus/tsar_Test/head_tsar2022_en_test_none.tsv"), 
                ("es", "../corpus/tsar_Test/head_tsar2022_es_test_none.tsv"), 
                ("pt", "../corpus/tsar_Test/head_tsar2022_pt_test_none.tsv")]:
    with open(ds) as inputfile:
        # lang = ds .split("/")[-1].split("_")[1]
        all_ds_tsar_final[lang] = []
        for ln in inputfile:
            cols = ln.strip().split("\t")
            sent = cols[0]
            target = cols[1]
            reply = Counter(cols[2:])
            assert target in sent
            assert len(sent.split("<head>"+target+"<head>")) == 2
            all_ds_tsar_final[lang].append({"sent": sent, "target": target, "ans": reply, "ds_name": ds})
        print(lang,len(all_ds_tsar_final[lang]))


# all_ds_tsar = {}
# for lang, ds in [("en", "data/test_corpus_tsar_en.tsv"), ("es", "data/test_corpus_tsar_es.tsv"), ("pt", "data/test_corpus_tsar_pt.tsv")]:
#     with open(ds) as inputfile:
#         # lang = ds .split("/")[-1].split("_")[1]
#         all_ds_tsar[lang] = []
#         for ln in inputfile:
#             cols = ln.strip().split("\t")
#             sent = cols[0]
#             target = cols[1]
#             reply = Counter(cols[2:])
#             assert target in sent
#             assert len(sent.split("<head>"+target+"<head>")) == 2
#             all_ds_tsar[lang].append({"sent": sent, "target": target, "ans": reply, "ds_name": ds})


# semeval_ds = {}
# for lang, path in [ ("en", "data/test_corpus_semeval_en.tsv"), ("es", "data/test_corpus_semeval_es.tsv"), ("pt", "data/test_corpus_semeval_pt.tsv")]:
#     semeval_ds[lang] = []
#     with open(path) as input_file:
#         for ln in input_file:
#             if "<head>" not in ln:
#                 continue
#             cols = ln.strip().split("\t")
#             # print(cols)
#             # input("...")
#             sent = cols[0]
#             ans = cols[2:]
#             if "<head>" not in sent or len(sent.split("<head>")) > 3:
#                 continue
#             target = sent.split("<head>")[1]
#             if target != cols[1]:
#                 ans = cols[1:]
#             # target = cols[1]
#             if target not in ans:
#                 ans.append(target)
#             # sent = sent.replace("<head>","")
#             semeval_ds[lang].append({"sent": sent, "target": target, "ans": ans, "gold": ans, "ds_name": path.split("/")[-1].replace(".csv","")})
#             # print(semeval_ds[lang][-1])
#             # input("...")
#         lst = semeval_ds[lang] 
#         lst = lst[int(len(lst)*.8):]
#         semeval_ds[lang] = lst
#     print(lang,len(semeval_ds[lang]))


####################
####
####    Query expansion
####
####################


# the binary FastText model should by in the TSAR-2022-Shared-Task-main diretory
# complex word is repeated |sent|/2
def __expansion(ds = [("tsar2022",None)], buffer_pattern = '_all_langs_fasttextPred_QE_'):
    for ds_name, all_ds in ds:
        for lang in all_ds:
            if os.path.exists('qe_' + ds_name + buffer_pattern +lang+'.txt'): ## the expansion is quite slow, so we buffer it
                print("loading pregenerated Query Expansion:", ds_name + '_all_langs_fasttextPred_QE_'+lang+'.txt')
                continue
            all_langs = {}
            tf_model = gensim.models.fasttext.load_facebook_model('/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/TSAR-2022-Shared-Task-main/cc.'+lang+'.300.bin') # too much memory for preloading
            print('Lang',lang)
            predictions = []
            ds_file = None
            results = {}
            for case in tqdm(all_ds[lang]):
                # print(case["sent"])
                lst = tf_model.wv.most_similar(positive=[case["target"]]*int(max(1,round(len(case["sent"].replace("<head>","").split(" "))/2,0))) + case["sent"].replace("<head>","").split(" "), negative=[], topn=5)
                lst = [w for w,s in lst]
                predictions.append({"sent":case["sent"],"target":case["target"], "ans":lst})
                ds_file = case["ds_name"]    
            all_langs[lang] = predictions
            tf_model = None # garbage collector doesn't work properly
            with open('qe_' + ds_name + buffer_pattern +lang+'.txt', 'w') as output_file:
                for lang in all_langs:
                    for cand in all_langs[lang]:
                        output_file.write(lang+"\t"+cand["sent"]+"\t"+cand["target"]+"\t"+"\t".join(cand["ans"]) + "\n")

 # method: 1 = repeat the entire sent ; 2 = use only the QE words without repeat the entire sent
def __qe_bert_generation(method = 2, buffer_pattern = '_all_langs_fasttextPred_QE_', ds_structure=[("semeval",None)], bert_models = {"es":"dccuchile/bert-base-spanish-wwm-cased","pt":"neuralmind/bert-base-portuguese-cased","en":"bert-large-uncased",}, max_length=512):
    ds = {}
    # all_langs = {"en":{},"es":{},"pt":{}}
    for file in glob.glob("*" + buffer_pattern+"*"+'.txt'):
        # ds_name = file.split("/")[-1].split("_")[0]+"_"+file.split("/")[-1].split("_")[1] # qe_semeval_all_langs_fasttextPred_QE_en.txt
        ds_name = file.split("/")[-1].split("_")[1] # qe_semeval_all_langs_fasttextPred_QE_en.txt
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
    print("--------")
    print(ds.keys())
    print("--------")
    ret = []
    for ds_name, all_ds in ds_structure:
        for lang in all_ds:
            if lang not in bert_models:
                continue
            ret.append(ds_name + "_" + lang + "_" + bert_models[lang].replace("/","_") +"_m" +str(method)+ ".txt")
            with open("qe"+ds_name + "_" + lang + "_" + bert_models[lang].replace("/","_") +"_m" +str(method) + "_lm" + bert_models[lang].replace("/","_") + ".txt","w") as output_file, open("qe"+ds_name + "_" + lang + "_" + bert_models[lang].replace("/","_") +"_m" +str(method) + "_lm" + bert_models[lang].replace("/","_") +  "_probs.txt","w") as output_prob_file:
                for multilanguage in [False]:
                    bad_scores = []
                    good_scores = []
                    print("*"*50)
                    print(ds_name, 'Lang', lang, "(M)" if multilanguage else "")
                    print("*"*50)
                    bert = bert_models["m" if multilanguage else lang]
                    if "bert-" in bert:
                        mask_tk = "[MASK]"
                        tokenizer = BertTokenizer.from_pretrained(bert)
                    else:
                        mask_tk = "<mask>"
                        tokenizer = RobertaTokenizer.from_pretrained(bert)

                    # mask_tk = "[MASK]" if "bert-" in bert else "<mask>"
                    bert = pipeline('fill-mask',model=bert, use_fast=True, tokenizer=tokenizer, top_k=20)
                    predictions = []
                    ds_file = None
                    results = {}
                    skips = 0
                    for case in tqdm(all_ds[lang]):
                        # print(case["sent"])    
                        num_context = 5
                        try_again = True
                        while try_again:
                            # try:
                            if True:
                                sequence_to_classify = case["sent"].replace("<head>","") + " " 
                                if method == 1:
                                    sequence_to_classify += " ".join( [ case["sent"].replace("<head>"+case["target"]+"<head>", w) for w in ds[ds_name][lang][case["sent"] +"\t" +case["target"] ][:num_context] ] )  
                                elif method == 2:
                                    sequence_to_classify += " ".join( [ w for w in ds[ds_name][lang][case["sent"] +"\t" +case["target"] ][:num_context] ] )  
                                sequence_to_classify = sequence_to_classify.replace("<head>", "")
                                
                                if "<head>"+case["target"]+"<head>" in case["sent"]:
                                    sequence_to_classify += " " + case["sent"].replace("<head>"+case["target"]+"<head>", mask_tk)
                                else:
                                    # sequence_to_classify += " " + case["sent"].replace(case["target"], mask_tk)
                                    # assert False
                                    skips+=1
                                    print("skip no <head>",skips, case["target"], "///", case["sent"])
                                    continue
                                tokenized_text = tokenizer.encode(sequence_to_classify, truncation = True, max_length = max_length)
                                while len(tokenized_text) > max_length-2:
                                    # print("tokenized_text", tokenized_text)
                                    # print("cut", sequence_to_classify)
                                    # print("new", " ".join(sequence_to_classify.split(" ")[1:]))
                                    sequence_to_classify = " ".join(sequence_to_classify.split(" ")[1:])
                                    tokenized_text = tokenizer.encode(sequence_to_classify, truncation = True, max_length = max_length)
                                    # input("...")

                                ans = bert(sequence_to_classify.replace("<head>", ""))
                                # lst = [x for _,x in sorted([(v["score"],v["token_str"]) for v in ans], reverse = True)]
                                lst = sorted([(v["score"],v["token_str"]) for v in ans], reverse = True)
                                output_prob_file.write(case["sent"].replace("<head>","") + "\t" + case["target"] + "\t" + "\t".join([w + "::"+str(s) for s, w in lst]) + "\n")

                                lst = [x for _,x in lst]
                                predictions.append({"sent":case["sent"].replace("<head>",""),"target":case["target"], "ans":lst})
                                ds_file = case["ds_name"]
                                output_file.write(case["sent"].replace("<head>","") + "\t" + case["target"] + "\t" + "\t".join(lst) + "\n")
                                try_again = False
                            # except:
                            #     print("skip",ds_name + "_" + lang + "_" + bert_models[lang].replace("/","_") +"_" +str(num_context)+ ".txt")     
                            #     num_context -= 1
    return ret

def query_expansion(ds = [("tsar2022",None)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 2, bert_models = {"es":"dccuchile/bert-base-spanish-wwm-cased","pt":"neuralmind/bert-base-portuguese-cased","en":"bert-large-uncased",}):
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

def double_sentence(use_sep = False, ds = [("tsar2022",None)], bert_models = None): 

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
                if "bert-" in model_name:
                    mask_tk = "[MASK]"
                    tokenizer = BertTokenizer.from_pretrained(model_name)
                else:
                    mask_tk = "<mask>"
                    tokenizer = RobertaTokenizer.from_pretrained(model_name)
                max_length = tokenizer.model_max_length
                output_file_name = "double_" + ds_name + "_" + lang + "_" + model_name.replace("/","_") +"_sep" +str(use_sep)+ ".txt"
                outwithscore_name = "score_"+output_file_name
                with open(output_file_name, "w") as output_file, open(outwithscore_name,"w") as output_file_with_score:
                    bert = pipeline('fill-mask',model=model_name, tokenizer=tokenizer, use_fast=True, top_k=20)
                    for case in tqdm(all_ds[lang]):
                        if use_sep:
                            if "bert-" in model_name:
                                sequence_to_classify = case["sent"].replace("<head>","").strip() + " [SEP] "+ case["sent"].replace("<head>"+case["target"]+"<head>", mask_tk)
                            else:
                                sequence_to_classify = case["sent"].replace("<head>","").strip() + "</s>" + "<s>" + case["sent"].replace("<head>"+case["target"]+"<head>", mask_tk)
                        else:
                            sequence_to_classify = case["sent"].replace("<head>","").strip() + " "+ case["sent"].replace("<head>"+case["target"]+"<head>", mask_tk)
                        
                        tokenized_text = tokenizer.encode(sequence_to_classify, truncation = True, max_length = max_length)
                        toks = tokenizer.convert_ids_to_tokens(tokenized_text)[1:-1]
                        if len(tokenized_text) > max_length-2:
                            toks = tokenizer.convert_ids_to_tokens(tokenized_text)
                            toks = toks[1:-1]# drop <s> and </s> / [CLS] and [SEP]
                            # toks = toks[:-1]# drop <s> and </s>
                            print([ [i, tk] for i, tk in enumerate(toks)])
                            pos_mask = [i for i, w in enumerate(toks) if w == mask_tk][0]
                            to_the_end = len(tokenized_text) #- pos_mask
                            to_the_begin = pos_mask -1
                            if to_the_end > to_the_begin and to_the_end > max_length/2: # the culprit is the last part
                                while to_the_end + to_the_begin +1 > max_length-2:
                                    to_the_end -= 1
                            elif to_the_begin > to_the_end and to_the_begin > max_length/2: # the culprit is the first part
                                while to_the_end + to_the_begin +1 > max_length-2:
                                    to_the_begin -= 1
                            else: 
                                while to_the_end + to_the_begin +1 > max_length-2:
                                    to_the_begin -= 1
                                    to_the_end -= 1
                            toks = toks[max_length-to_the_begin:] + toks[pos_mask:to_the_end]
                            if "bert-":
                                sequence_to_classify = " ".join(toks).replace(' ##', '')
                            else:
                                sequence_to_classify = "".join(toks).replace('??', ' ')
                        ans = bert(sequence_to_classify)
                        # print(ans[0].keys())
                        lst = [(v["score"],v["token_str"]) for v in ans]
                        # print(lst[:3])
                        lst = sorted(lst, reverse = True)
                        lst_score = [w + "::"+str(s) for s, w in lst]
                        lst = [w.strip() for s, w in lst]
                        output_file.write(case["sent"].replace("<head>","") + "\t" + case["target"]  + "\t" + "\t".join(lst) + "\n")
                        output_file_with_score.write(case["sent"].replace("<head>","") + "\t" + case["target"]  + "\t" + "\t".join(lst_score) + "\n")
                        # predictions[model_name].append({"sent":case["sent"],"target":case["target"], "gold":case["gold"], "ans":lst})

####################
####
####    Paraphrases
####
####################

def __predict_masked_sent(text, tokenizer, model, masks, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    #print(" ".join(tokenized_text))
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
    predictions = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        predictions.append((token_weight, predicted_token))
        masks.write("[MASK]:\t"+predicted_token+"\t weights:"+str(float(token_weight))+"\n")
    return predictions

def __paraphrase(text, num_return_sequences, num_beams, tokenizer, model, max_length=128):
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids, num_return_sequences=num_return_sequences, num_beams=num_beams, max_length=max_length, no_repeat_ngram_size=2, repetition_penalty=3.5, length_penalty=1.0, early_stopping=True)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds

def __get_response(input_text,num_return_sequences,num_beams, ptokenizer, pmodel, max_length=512):
    #batch = ptokenizer(input_text,truncation=True,max_length=1000, return_tensors="pt").to(torch_device)
    batch = ptokenizer(input_text, truncation=True, max_length=max_length, return_tensors="pt")
    # print(len(input_text),len(batch), len(batch["input_ids"][0]), batch)
    # print(input_text)
    translated = pmodel.generate(**batch, max_length=max_length, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=30)
    #translated = pmodel.generate(**batch)
    tgt_text = ptokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def alternatives(ds = [("tsar2022",None)], output_name="resultats/",  bert_models = None, models_para=None):
    #tuner007/pegasus_paraphrase
    for ds_name, all_ds in ds:
        for lang in all_ds:
            print('Lang',lang)
            tokenizer = BertTokenizer.from_pretrained(bert_models[lang])
            model = BertForMaskedLM.from_pretrained(bert_models[lang])
            model.eval()

            model_name = models_para[lang]
            torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if "pegasus-" in model_name:
                ptokenizer = PegasusTokenizer.from_pretrained(model_name)
                pmodel = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
            else:
                ptokenizer = BertTokenizer.from_pretrained(bert_models[lang])
                pmodel = AutoModel.from_pretrained(model_name)
            
            if not os.path.exists(output_name):
                    os.mkdir(output_name)
            print("WARNING", output_name, "already exists")

            num_beams = 8
            num_return_sequences = 8
            #f = open("data/"+ds_name+"_"+lang+"_trial_none.tsv",'r')
            #print("data/"+ds_name+"_"+lang+"_trial_none.tsv")
            out = open(output_name+ds_name+"_"+str(num_beams)+"_"+str(num_return_sequences)+"_"+lang+".tsv","w")
            out_probs = open(output_name+ds_name+"_"+str(num_beams)+"_"+str(num_return_sequences)+"_"+lang+"_probs.tsv","w")
            masks = open("masks/"+ds_name+"_"+str(num_beams)+"_"+str(num_return_sequences)+"_"+lang+".txt",'w')
            for case in tqdm(all_ds[lang]):
                sent = case["sent"]
#                sent,word = l.rstrip().split("\t")
                word = case["target"]
                sout = sent.replace("<head>","")
                srep = sent.replace("<head>"+word+"<head>","[MASK]")
                context = srep
                #masks.write(l)
                paraphrases = []
                if "pegasus-" in lang:
                    paraphrases = __get_response(context,num_return_sequences,num_beams, ptokenizer=ptokenizer, pmodel=pmodel)
                else:
                    paraphrases = __paraphrase(context,num_return_sequences,num_beams, tokenizer=tokenizer, model=model)           
                for p in paraphrases:
                    #print(context+" "+str(len(tokenizer.tokenize(context))+len(tokenizer.tokenize(p))+len(tokenizer.tokenize(srep))))
                    if len(tokenizer.tokenize(context))+len(tokenizer.tokenize(p))+len(tokenizer.tokenize(srep)) <= 500:
                        context += " "+p
                    else:
                        break
                context += " "+srep
                #print(context)
                pred = __predict_masked_sent(context, top_k=20, tokenizer=tokenizer, model=model, masks=masks)
                predfilt = []
                predfilt_prob = []
                for w, p in pred: #(token_weight, predicted_token)
                    # x = re.search("^[a-zA-Z].*[a-zA-Z]$",p)
                    # if word not in p and x != None:
                    predfilt.append(p)
                    predfilt_prob.append(p + "::" + str(w))
                final = sout+"\t"+word+"\t"+"\t".join(predfilt)+"\n"
                #print(final)
                out.write(final)
                out_probs.write(sout+"\t"+word+"\t"+"\t".join(predfilt_prob)+"\n")
           # f.close()
            out.close()
            masks.close()

    
####################
####
####    Execution
####
####################


bert_models = {"es":[
                        # "skimai/spanberta-base-cased", 
                        # "PlanTL-GOB-ES/roberta-base-bne",
                        "dccuchile/bert-base-spanish-wwm-cased",    #bert
                        "dccuchile/bert-base-spanish-wwm-uncased"    #bert
                     
                    ],
               "pt":[
                        # "josu/roberta-pt-br",
                        # "rdenadai/BR_BERTo",
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



# double_sentence(use_sep = False, ds = [("tsar2022final",all_ds_tsar_final)], bert_models = bert_models)
# # double_sentence(use_sep = True, ds = [("tsar2022final",all_ds_tsar_final)], bert_models = bert_models)

bert_models = {"es":"dccuchile/bert-base-spanish-wwm-cased","pt":"neuralmind/bert-large-portuguese-cased","en":"bert-large-uncased"} # run 1
query_expansion(ds = [("tsar2022final",all_ds_tsar_final)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 1, bert_models=bert_models)
query_expansion(ds = [("tsar2022final",all_ds_tsar_final)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 2, bert_models=bert_models)
# bert_models = {"pt":"neuralmind/bert-base-portuguese-cased","en":"bert-base-uncased"} # run 2
# query_expansion(ds = [("semeval", semeval_ds), ("tsar2022",all_ds_tsar)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 1, bert_models=bert_models)
# query_expansion(ds = [("tsar2022_final",all_ds_tsar_final)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 2, bert_models=bert_models)
bert_models = {"es":"skimai/spanberta-base-cased", "pt":"rdenadai/BR_BERTo","en":"roberta-large"} 
bert_models = {"en":"roberta-large"} 
query_expansion(ds = [("tsar2022final",all_ds_tsar_final)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 1, bert_models=bert_models)
query_expansion(ds = [("tsar2022final",all_ds_tsar_final)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 2, bert_models=bert_models)




# double_sentence(use_sep = False, ds = [("semeval", semeval_ds), ("tsar2022",all_ds_tsar)], bert_models = bert_models)
# double_sentence(use_sep = True, ds = [("semeval", semeval_ds), ("tsar2022",all_ds_tsar)], bert_models = bert_models)

# bert_models = {"es":"dccuchile/bert-base-spanish-wwm-cased","pt":"neuralmind/bert-base-portuguese-cased","en":"bert-large-uncased"} # run 1
# query_expansion(ds = [("semeval", semeval_ds), ("tsar2022",all_ds_tsar)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 1, bert_models=bert_models)
# query_expansion(ds = [("semeval", semeval_ds), ("tsar2022",all_ds_tsar)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 2, bert_models=bert_models)
# bert_models = {"pt":"neuralmind/bert-large-portuguese-cased","en":"bert-base-uncased"} # run 2
# query_expansion(ds = [("semeval", semeval_ds), ("tsar2022",all_ds_tsar)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 1, bert_models=bert_models)
# query_expansion(ds = [("semeval", semeval_ds), ("tsar2022",all_ds_tsar)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 2, bert_models=bert_models)
# bert_models = {"es":"skimai/spanberta-base-cased", "pt":"rdenadai/BR_BERTo","en":"roberta-large"} 
# query_expansion(ds = [("semeval", semeval_ds), ("tsar2022",all_ds_tsar)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 1, bert_models=bert_models)
# query_expansion(ds = [("semeval", semeval_ds), ("tsar2022",all_ds_tsar)], buffer_pattern = '_all_langs_fasttextPred_QE_', method = 2, bert_models=bert_models)



# bert_models = {"en":"bert-large-uncased", "es":"dccuchile/bert-base-spanish-wwm-cased","pt":"neuralmind/bert-large-portuguese-cased"}
# models_para = {"en":"google/mt5-base","es":"google/mt5-base",      "pt":"google/mt5-base"}
# alternatives(ds = [("tsar2022_final",all_ds_tsar_final)], output_name="resultats_mt5/", bert_models=bert_models, models_para=models_para)

# bert_models = {"en":"bert-large-uncased", "es":"dccuchile/bert-base-spanish-wwm-cased","pt":"neuralmind/bert-large-portuguese-cased"}
# models_para = {"en":"google/pegasus-xsum","es":"seduerr/mt5-paraphrases-espanol",      "pt":"unicamp-dl/ptt5-base-portuguese-vocab"}
# alternatives(ds = [("tsar2022_final",all_ds_tsar_final)], output_name="resultats_paraphrase/", bert_models=bert_models, models_para=models_para)


print("done !")
