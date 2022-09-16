import glob
import os
import re
import stanza
from tqdm import tqdm

def fix_espaces(sent, complex, cands, probs):
    new_cands = []
    for c in cands:
        new_cands.append(c.replace(" ", ""))
    return sent, complex, new_cands, probs

def fix_case_and_non_words(sent, complex, cands, probs):
    new_cands, new_probs = [], []
    for c, p in zip(cands, probs):
        c = c.lower()
        x = re.search("^[a-zA-Z].*[a-zA-Z]$",c)
        if x != None:
            new_cands.append(c)
            new_probs.append(p)
    return sent, complex, new_cands, new_probs


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
                return [word.text, word.lemma, word.upos, word.xpos, word.feats]
    print(":(")
    print(target)
    print(s)
    return None




folders = [
            # # "/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/git/tsar-shared-task-2022/evaluation/double_sentence/results",
            #  "/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/git/tsar-shared-task-2022/evaluation/double_sentence/probs",
            # # "/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/git/tsar-shared-task-2022/evaluation/qe_output/results_QE",
            # "/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/git/tsar-shared-task-2022/evaluation/qe_output/probs",
            # "/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/git/tsar-shared-task-2022/evaluation/paraphrase/results_lang_specific/results",
            # "/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/git/tsar-shared-task-2022/evaluation/paraphrase/results_lang_specific/probs",
            # "/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/git/tsar-shared-task-2022/evaluation/paraphrase/multilang/probs",
            # "/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/git/tsar-shared-task-2022/evaluation/paraphrase/multilang/results",
            "results/results",
            "results/probs"
            ]


def __load_ds(path, process_probs):
    ds = []
    with open(path) as input_file:
        for ln in input_file:
            if "\t" not in ln:
                continue
            sent, complex, *cands = ln.strip().split("\t")

            if process_probs:
                probs = []
                new_cands = []
                for c in cands:
                    if ":::" in c:
                        index = c.rindex("::")
                        p = c[index+2:]
                        c = c[:index]
                    else:
                        c, p = c.split("::")
                    p = p.replace("tensor(","").replace(")","")
                    p = float(p)
                    new_cands.append(c)
                    probs.append(p)
                cands = new_cands
            else:
                probs = [2 for c in cands] # for simplicity we provide fake values and ignore them
            ds.append([sent, complex, cands, probs])
    return ds

# phrases = {}
# ds = [("tsar2022",all_ds_tsar),("semeval",semeval_ds)]
# for ds_name, all_ds in ds:
#         for lang in all_ds:
#             for case in all_ds[lang]:
#                 phrases[case["sent"].replace("<head>","").strip()] = case["sent"].strip()

##################""
def fix_agreement(folder, output_folder):
    golds = ["../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv",
            "../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv",
            "../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv",
            "../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv",
            "../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv",
            "../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv",
            "../corpus/tsar_Test/head_tsar2022_en_test_none.tsv",
            "../corpus/tsar_Test/head_tsar2022_es_test_none.tsv",
            "../corpus/tsar_Test/head_tsar2022_pt_test_none.tsv",]
    parsers = {"en": stanza.Pipeline(lang="en", processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True),
                "es": stanza.Pipeline(lang="es", processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True),
                "pt": stanza.Pipeline(lang="pt", processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True)
                }
    all_files_to_process = glob.glob(folder+"/*.txt")
    print("folder",folder)
    for index_all_files_to_process, f in enumerate(all_files_to_process):
        print("~~~> ", index_all_files_to_process, len(all_files_to_process), f)
        gold = None
        if "_semeval_" in f:
            if "_en_" in f:
                gold = golds[0]
            elif "_es_" in f:
                gold = golds[2]
            elif "_pt_" in f:
                gold = golds[4]
        elif "_tsar_" in f or "_tsar2022_" in f:
            if "_en_" in f:
                gold = golds[1]
            elif "_es_" in f:
                gold = golds[3]
            elif "_pt_" in f:
                gold = golds[5]
        elif "_tsar2022final_" in f:
            if "_en_" in f:
                gold = golds[6]
            elif "_es_" in f:
                gold = golds[7]
            elif "_pt_" in f:
                gold = golds[8]
        phrases = {}
        assert gold
        ds = __load_ds(gold, process_probs=False)
        print("gold", len(ds), gold)
        for sent, complex, cands, probs in ds:
            phrases[sent.replace("<head>","").strip()] = sent.strip()

        lines = open(f).readlines()
        lang = re.findall("_.._",f)[0].replace("_","")
        dela = __load_dela(lang)
        dela_flex = __load_delaflex(lang)
        nlp = parsers[lang]  # stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True)
        output_file_name = output_folder +"/"+ f.split("/")[-1] # "flex_"+f.split("/")[1]
        if os.path.exists(output_file_name):
            print("skip")
            continue
        output_file = open(output_file_name,"w")
        for ln in tqdm(lines):
            ln = ln.strip().split("\t")
            sent = ln[0].strip()
            # print(">>>>>", sent)
            cword = ln[1]
            if "<head>" in sent:
                if sent.split("<head>")[1] != cword:
                    print("Warning: cword changed from",cword, "to",sent.split("<head>")[1], "in", sent.split("<head>"))
                    cword = sent.split("<head>")[1]
            # for k in phrases:
            #     print(":::",k)
            if (sent in phrases and cword not in phrases[sent]) or cword not in phrases[sent.replace("<head>"," ")]:
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
                # print("##>",cword,phrases[sent])
                if "<head>" in phrases[sent] and phrases[sent].split("<head>")[1]!=cword:
                    print("Warning: cword changed from",cword, "to",phrases[sent].split("<head>")[1], "in", phrases[sent].split("<head>"))
                    to_analyze = phrases[sent].replace("<head>"+phrases[sent].split("<head>")[1]+"<head>"," "+c+" ")
                else:
                    to_analyze = phrases[sent].replace("<head>"+cword+"<head>"," "+c+" ")
                # print("##",cword,to_analyze, c)

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
##################""

for folder in folders:
    if not os.path.exists(folder + "_filter"):
        os.mkdir(folder + "_filter")
    process_probs = True if "/probs" in folder else False
    for file in glob.glob(folder + "/*"):
        # with open(file) as input_file, open(folder + "_filter/" + file.split("/")[-1], "w") as output_file:
        with open(folder + "_filter/" + file.split("/")[-1], "w") as output_file:
            ds = __load_ds(file, process_probs=process_probs)
            # for ln in input_file:
            #     if "\t" not in ln:
            #         continue
            #     sent, complex, *cands = ln.strip().split("\t")

            #     if process_probs:
            #         probs = []
            #         new_cands = []
            #         for c in cands:
            #             if ":::" in c:
            #                 index = c.rindex("::")
            #                 p = c[index+2:]
            #                 c = c[:index]
            #             else:
            #                 c, p = c.split("::")
            #             p = p.replace("tensor(","").replace(")","")
            #             p = float(p)
            #             new_cands.append(c)
            #             probs.append(p)
            #         cands = new_cands
            #     else:
            #         probs = [2 for c in cands] # for simplicity we provide fake values and ignore them

            #     ##########
            for sent, complex, cands, probs in ds:
                ## real stuff
                sent, complex, cands, probs = fix_espaces(sent, complex, cands, probs)
                sent, complex, cands, probs = fix_case_and_non_words(sent, complex, cands, probs)


                ##########
                if process_probs:
                    cands = [c+"::"+str(p) for c, p in zip(cands, probs)]
                output_file.write(sent+"\t"+complex+"\t"+"\t".join(cands) + "\n")

    if process_probs:
       if not os.path.exists(folder + "_agree"):
        os.mkdir(folder + "_agree")
       fix_agreement(folder + "_filter", folder + "_agree")
