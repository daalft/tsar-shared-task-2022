#!/usr/bin/env python

import stanza
import json
from jsonmerge import merge
from collections import Counter


all_ds_tsar = {}
for ds in ["data/tsar2022_en_test_none.tsv",
           "data/tsar2022_es_test_none.tsv",
           "data/tsar2022_pt_test_none.tsv",
           "data/tsar2022_en_trial_gold.tsv",
           "data/tsar2022_es_trial_gold.tsv",
           "data/tsar2022_pt_trial_gold.tsv"
           ]:
    with open(ds) as inputfile:
        lang = ds .split("/")[-1].split("_")[1]
        if lang not in all_ds_tsar:
            all_ds_tsar[lang] = []
        for ln in inputfile:
            cols = ln.strip().split("\t")
            sent = cols[0]
            target = cols[1]
            reply = Counter(cols[2:])
            assert target in sent
            assert len(sent.split("<head>"+target+"<head>")) == 2
            all_ds_tsar[lang].append({"sent": sent, "target": target, "ans": reply, "ds_name": ds})


semeval_ds = {}
for lang, path in {"en": "data/semeval_en.tsv", "es": "data/semeval_es.tsv"}.items():
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
    #print(lang,len(semeval_ds[lang]))



dela_all = {
    "en":"dico/dela-en.dic",
    "es":"dico/dela-es.dic",
    "pt":"dico/dela-pt.dic"
}

def load_dela(lang):
    dic = {}
    with open(dela_all[lang]) as file:
        for ln in file:
            l = ln.strip()
            if (lang == "pt") and ("V+PRO" in l):
                continue
            #print(l)
            if len(l.replace("\.","").split(".")) == 2: # écarte le souci des abréviations de mois en espagnol (abr., oct., nov., ...) :
                key, value = l.replace("\.","").split(".")
                if len(value.split(":")) > 2:
                    values = []
                    flexions = value.split(":")
                    for f in flexions:
                        if f != flexions[0]:
                            values.append(flexions[0]+":"+f)
                    for v in values:
                        if key in dic:
                            dic[key].append(v)
                        else:
                            dic[key] = [v]
                if key in dic:
                    dic[key].append(value)
                else:
                    dic[key] = [value]
        return dic
    
def load_delaflex(lang):
    dic = {}
    with open(dela_all[lang]) as file:
        for ln in file:
            l = ln.strip()
            if (lang == "pt") and ("V+PRO" in l):
                continue
            value, key = l.split(",")
            if len(value.split(":")) > 2:
                keys = []
                flexions = key.split(":")
                for f in flexions:
                    if f != flexions[0]:
                        keys.append(flexions[0]+":"+f)
                for k in keys:
                    if key in dic:
                        dic[key].append(v)
                    else:
                        dic[key] = [v]
            
            if key in dic:
                dic[key].append(value)
            else:
                dic[key] = [value]
        return dic
    
def analyse_ms(ph,nlp):
    s = ph["sent"].replace("<head>"," ")
    ann = nlp(s)
    for sent in ann.sentences:
        cword = ph["target"]
        for word in sent.words:
            if word.text == cword:
                return word.text, word.lemma, word.upos, word.xpos, word.feats
                #print("word:",word.text,"\tlemma:",word.lemma,"\tupos:",word.upos,"\txpos:", word.xpos,"\tfeats:",word.feats)
              
def load_dbnary(pos,lang):
    fhyper = open("dbnary/no_synset/dbnary_"+lang+"_hypernym_"+pos+".json")
    fhypo = open("dbnary/no_synset/dbnary_"+lang+"_hyponym_"+pos+".json")
    fsyno = open("dbnary/no_synset/dbnary_"+lang+"_synonym_"+pos+".json")
    hyper = json.load(fhyper)
    hypo = json.load(fhypo)
    syno = json.load(fsyno)
    fhyper.close()
    fhypo.close()
    fsyno.close()
    m = merge(hyper,hypo)
    return merge(m,syno)
    
    
def dico(ds = [("tsar2022",all_ds_tsar),("semeval",semeval_ds)]):
    for ds_name, all_ds in ds:
            for lang in all_ds:
                output_file = open("dico_"+ds_name+"_"+lang+".tsv",'w')
                count = 0
                countok = 0
                print('Lang',lang)
                nlp = stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True)
                dela = load_dela(lang)
                dela_flex = load_delaflex(lang)
                dbnary = {}
                for pos in "adjective","noun","adverb","verb":
                    dbnary[pos] = load_dbnary(pos,lang)
                for case in all_ds[lang]:
                    count += 1
                    if " " in case["target"]:
                        output_file.write(case["sent"].replace("<head>","")+"\t"+case["target"]+"\t"+case["target"]+"\n")
                        continue
                    else:
                        word, lemma, upos, xpos, feats = analyse_ms(case,nlp)
                        word = word.lower()
                    if lemma in dbnary[stanza_to_dbnary[upos]]:
                        countok += 1
                        cand = dbnary[stanza_to_dbnary[upos]][lemma]
                        #print("\t".join(cand))
                        if (word == lemma):
                            if word+"," in dela:
                                flex = dela[word+","]
                            else:
                                output_file.write(case["sent"].replace("<head>","")+"\t"+case["target"]+"\t"+case["target"]+"\n")
                                continue
                        else:
                            flex = dela[word+","+lemma]
                        candok = []
                        for c in cand:
                            for f in flex:
                                if c+"."+f in dela_flex:
                                    candok.append(dela_flex[c+"."+f])
                        if len(candok) > 0:
                            candout = []
                            for ok in candok:
                                #print(ok[0])
                                if ok[0] not in candout:
                                    candout.append(ok[0])
                            output_file.write(case["sent"].replace("<head>","")+"\t"+case["target"]+"\t"+case["target"]+"\t"+"\t".join(candout)+"\n")
                        else:
                            output_file.write(case["sent"].replace("<head>","")+"\t"+case["target"]+"\t"+case["target"]+"\n")
                    else:
                        cand = []
                        output_file.write(case["sent"].replace("<head>","")+"\t"+case["target"]+"\t"+case["target"]+"\n")
                        #print(lemma,"not found in dbnary","\t".join(cand))
                #print(lang,str(countok/count*100))
                output_file.close()
                    
stanza_to_dbnary = {
    "ADJ":"adjective",
    "NOUN":"noun",
    "PROPN":"noun",
    "ADV":"adverb",
    "VERB":"verb"
}
dico()       
