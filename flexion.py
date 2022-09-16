#!/usr/bin/env python

import glob
from collections import Counter
import stanza
import re

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
for lang, path in {"en": "data/semeval_en.tsv", "es": "data/semeval_es.tsv", "pt":"data/semeval_pt.tsv"}.items():
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

stanza_to_dela = {
    "ADJ":"A",
    "ADP":"A",
    "NOUN": "N",
    "PROPN":"N",
    "ADV":"ADV",
    "VERB":"V"
}

def __load_dela(lang):
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
        

    
def __load_delaflex(lang):
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
    
def __analyse_ms(s,target,nlp):
    ann = nlp(s)
    for sent in ann.sentences:
        for word in sent.words:
            word.text = word.text.replace(")","")
            word.text = word.text.replace(".","")
            #print(word.text,target)
            if word.text == target:
                return word.text, word.lemma, word.upos, word.xpos, word.feats




phrases = {}
ds = [("tsar2022",all_ds_tsar),("semeval",semeval_ds)]
for ds_name, all_ds in ds:
        for lang in all_ds:
            for case in all_ds[lang]:
                phrases[case["sent"].replace("<head>","").strip()] = case["sent"].strip()
                
for f in glob.glob("probs/*.txt"):
    lines = open(f).readlines()
    lang = re.findall("_.._",f)[0].replace("_","")
    dela = __load_dela(lang)
    dela_flex = __load_delaflex(lang)
    nlp = stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True)
    output_file_name = "flex_"+f.split("/")[1]
    output_file = open(output_file_name,"w")
    for ln in lines:
        ln = ln.strip().split("\t")
        sent = ln[0].strip()
        cword = ln[1]
        if cword not in phrases[sent]:
            output_file.write("\t".join(ln)+"\n")
            continue
        out = [sent,cword]
        flexed_candidates = {}
        if " " in cword:
            output_file.write("\t".join(ln)+"\n")
            continue
        word, lemma, upos, xpos, feats = __analyse_ms(phrases[sent].replace("<head>"," "),cword,nlp)
        word = word.lower()
        flex = []
        if (word == lemma and lang != "pt"):
            if word+"," in dela:
                flex = [entry for entry in dela[word+","] if entry.split(":")[0] == stanza_to_dela[upos]]
            else:
                output_file.write("\t".join(ln)+"\n")
                continue
        else:
            if word+","+lemma in dela:
                flex = [entry for entry in dela[word+","+lemma] if entry.split(":")[0] == stanza_to_dela[upos]]
            else:
                output_file.write("\t".join(ln)+"\n")
                continue
        for candidate in ln[2:]:
            c, prob = candidate.split("::")
            x = re.search("^[a-zA-Z].*[a-zA-Z]$",c)
            if x == None:
                continue
            to_analyze = phrases[sent].replace("<head>"+cword+"<head>"," "+c+" ")
            #print(cword,to_analyze)
            sword, slemma, supos, sxpos, sfeats = __analyse_ms(to_analyze,c,nlp)
            sword = sword.lower()
            sflex = ""
            if (sword == slemma):
                if sword+"," in dela:
                    sflex = dela[sword+","]
                #else:
                 #   if c in flexed_candidates:
                 #       flexed_candidates[c] += prob
                 #   else:
                 #       flexed_candidates[c] = prob
                 #       continue
            else:
                if word+","+lemma in dela:
                    sflex = dela[word+","+lemma]
                else:
                    if c in flexed_candidates:
                        flexed_candidates[c] += prob
                    else:
                        flexed_candidates[c] = prob
                    continue
            flex_found = False
            flexok = ""
            for i in flex:
                for j in sflex:
                    if i == j:
                        flex_found = True
                        flexok = i
                        break
            if flex_found == True:
                if c in flexed_candidates:
                    flexed_candidates[c] += prob
                #else:
                #    flexed_candidates[c] = prob
                continue
            else:
                if c+"."+flexok in dela_flex:
                    flexed_c = dela_flex[flexok]
                else:
                    flexed_c = c
                if flexed_c in flexed_candidates:
                    flexed_candidates[flexed_c] += prob
                else:
                    flexed_candidates[flexed_c] = prob
        output_file.write(sent+"\t")
        output_file.write(cword+"\t")
        for k in flexed_candidates:
            output_file.write("\t"+k+"::"+flexed_candidates[k])
        output_file.write("\n")
            #print(c,prob)
            
    output_file.close()
    
    


