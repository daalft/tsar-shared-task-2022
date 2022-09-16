from collections import Counter
import numpy as np
from sklearn.metrics import jaccard_score
import glob

from ranx import Qrels, Run, evaluate, compare

def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def ark(predicted, actual, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def load_semeval_gold(path = "semeval_es_dccuchile_bert-base-spanish-wwm-cased_m2.txt"):
    semeval_pred = {}
    with open(path) as input_file:
        for ln in input_file:
            if "<head>" not in ln:
                continue
            cols = ln.strip().split("\t")
            sent = cols[0].replace("<head>","")
            complex = cols[0].split("<head>")[1]
            cands = cols[1:]
            cands = [c.strip().lower() for c in cands]
            semeval_pred[sent + "\t" + complex] = cands
    return semeval_pred

def load_semeval_pred(path = "semeval_es_dccuchile_bert-base-spanish-wwm-cased_m2.txt"):
    semeval_pred = {}
    with open(path) as input_file:
        for ln in input_file:
            cols = ln.replace("<head>","").strip().split("\t")
            sent = cols[0]
            complex = cols[1]
            cands = cols[2:]
            cands = [c.strip().lower() for c in cands]
            for i, c in enumerate(cands):
                if "::" in c:
                    cands[i] = c.split("::")[0]

            semeval_pred[sent + "\t" + complex] = cands
    return semeval_pred


def load_tsar_pred(path="tsar_pt_neuralmind_bert-base-portuguese-cased_m2.txt"):
    return load_semeval(path=path)

def load_tsar_gold(path="TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_en_trial_gold.tsv"):
    gold = {}
    with open(path) as input_file:
        for ln in input_file:
            cols = ln.replace("<head>","").strip().split("\t")
            sent = cols[0]
            complex = cols[1]
            cands = [(c,w)for c, w in Counter(cols[2:]).most_common()]
            cands = sorted(cands)
            cands = [w for c,w in cands]
            # cands = [w for c, w in Counter(cols[2:]).most_common()]
            gold[sent + "\t" + complex] = cands
        return gold


# def eval(ds1="semeval2012T1/2010_es.csv", ds2="semeval_es_dccuchile_bert-base-spanish-wwm-cased_m1.txt"):
#     # gold = load_semeval_gold("semeval2012T1/2012.csv")
#     # pred = load_semeval_pred("semeval_en_bert-large-uncased_m2.txt")
#     gold = load_semeval_gold(ds1)
#     pred = load_semeval_pred(ds2)
#     print("gold",len(gold), "pred", len(pred))
#     lst_apk = [1, 3, 5, 10]
#     # lst_apk = {1:[], 3:[], 5:[], 10:[]}
#     # lst_ark = {1:[], 3:[], 5:[], 10:[]}
#     # for key in pred:
#     #     cands_pred = pred[key]
#     #     cands_gold = gold[key]
#     #     for k in lst_apk:
#     #         lst_apk[k].append(apk(cands_gold,cands_pred,k=k))
#     #         lst_ark[k].append(ark(cands_gold,cands_pred,k=k))
#     gold_aux = {}
#     for s, cands in gold.items():
#         if s in pred:
#             gold_aux[s] = {}
#             for i, c in enumerate(cands):
#                 gold_aux[s][c] = i
#     pred_aux = {}
#     for s, preds in pred.items():
#         pred_aux[s] = {}
#         for i, c in enumerate(preds):
#             pred_aux[s][c] = i
#     qrels = Qrels(gold_aux)
#     run = Run(pred_aux)

#     jaccard_reuslts = {}
#     for sent in pred_aux:
#         pred_cands = pred_aux[sent]
#         gold_cands = gold_aux[sent]
#         for k in lst_apk:
#             j = jaccard(gold_cands, pred_cands, k)
#             jaccard_reuslts[k].append(j)
#     for k in jaccard_reuslts:
#         print(k, np.mean(jaccard_reuslts[k]))


#     metrics = ["hits", "hit_rate", "precision", "recall", "f1", "mrr", "map", "ndcg", "ndcg_burges"]
#     for metric in []:
#         for i in lst_apk:
#             metrics.append(m+"@"+str(i))
#     metrics.append("r-precision")
#     metrics.append("bpref")
#     score_dict = evaluate(qrels, run, metrics) # ["ndcg@3", "map@5", "f1@1","map@1", "mrr"]
#     print(score_dict)
#     for k in lst_apk:
#         print(k, np.mean(lst_apk[k]), np.mean(lst_ark[k]))
#     # map3 = np.mean([apk(a,p,k=3) for a,p in zip(all_gold, all_pred)])
#     # print(map3)

def jaccard(gold, pred, k):
    # print(min(k,len(gold)), k, len(gold), gold)
    gold = set(gold[:min(k,len(gold))])
    pred = set(pred[:min(k,len(pred))])
    i = gold & pred
    u = gold | pred
    return len(i)/len(u)

def jaccardd(gold, pred, k):
    # print(min(k,len(gold)), k, len(gold), gold)
    gold = set(gold[:min(k,len(gold))])
    pred = set(pred[:min(k,len(pred))])
    i = gold.symmetric_difference(pred)
    u = gold | pred
    return len(i)/len(u)

def save_results(file_name, report, p_value=0.01, sep = "\t"):
    metrics = report["metrics"]
    model_names = report["model_names"]
    # for k in report:
    #   print(k, report[k])
    # print("report.comparisons",report.comparisons)
    with open(file_name, "w") as output_file:
        output_file.write("ID"+sep)
        output_file.write("Input file"+sep)
        for m in metrics:
            output_file.write(m + sep + m+" (pvalue<"+ str(p_value) + ")"+ sep)
        output_file.write("\n")
        
        for model_name in model_names:
            output_file.write(model_name + sep)
            output_file.write(report["run_names"][model_name] + sep)
            for metric in metrics:
                output_file.write(str(round(report[model_name]['scores'][metric],4)) + sep)
                diff_models = [model for model in report[model_name]["comparisons"] if metric in report[model_name]["comparisons"][model] and report[model_name]["comparisons"][model][metric] < p_value and report[model_name]['scores'][metric] > report[model]['scores'][metric]]
                output_file.write(";".join(diff_models) + sep)
            output_file.write("\n")



def eval_lst(file_name, ds1="semeval2012T1/2010_es.csv", ds2=["semeval_es_dccuchile_bert-base-spanish-wwm-cased_m1.txt"], lst_apk = [1, 3, 5 ,10], runjaccard2=False):
    # gold = load_semeval_gold("semeval2012T1/2012.csv")
    # pred = load_semeval_pred("semeval_en_bert-large-uncased_m2.txt")
    gold = load_semeval_gold(ds1)
    preds_k = set()
    preds = []
    for ds in ds2:
        pred = load_semeval_pred(ds)
        preds.append(pred)
        for k in pred:
            preds_k.add(k)

    # pred = load_semeval_pred(ds2)
    print("gold",len(gold), "runs", len(preds), "preds:", [len(r) for r in preds])
    for i, ds in enumerate(ds2):
        print("\t", len(preds[i]), ds)
    

    gold_aux = {}
    for s, cands in gold.items():
        if s in preds_k:
            gold_aux[s] = {}
            for i, c in enumerate(cands):
                gold_aux[s][c] = i
    assert len(gold_aux)>0
    
    runs = []
    drop_k = []
    for pred in preds:
        pred_aux = {}
        for s, p in pred.items():
            if s in preds_k and s in gold_aux:
                pred_aux[s] = {}
                for i, c in enumerate(p):
                    pred_aux[s][c] = i
            elif s in gold_aux:
                drop_k.append(s)
        run = Run(pred_aux)
        runs.append(run)
    
    for k in drop_k:
        del gold_aux[k]
    qrels = Qrels(gold_aux)

    print("...gold",len(gold_aux), "runs", len(pred_aux), "preds:", [len(r) for r in runs])
    ###############################################################
    metrics = []
    for m in ["hits", "hit_rate", "precision", "recall", "f1", "mrr", "map", "ndcg", "ndcg_burges"]:
        for i in lst_apk:
            metrics.append(m+"@"+str(i))
    metrics.append("r-precision")
    metrics.append("bpref")
    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=metrics,
        stat_test="student",
        max_p=0.05  # P-value threshold
    ).to_dict()
    report["run_names"] = {}
    for index, _ in enumerate(preds):
        report["run_names"]["run_" + str(index+1)] = ds2[index]
    ###############################################################
    jaccard_reuslts = {k:{} for k in lst_apk}
    jaccardd_reuslts = {k:{} for k in lst_apk}
    for k in lst_apk:
        report["metrics"].append("Jaccard@" + str(k))
        report["metrics"].append("Jaccard_dist@" + str(k))
    for index, pred_aux in enumerate(preds):
        for sent in pred_aux:
            if sent not in gold_aux:
                continue
            pred_cands = pred_aux[sent]
            gold_cands = gold_aux[sent]
            for k in lst_apk:
                j = jaccard(list(gold_cands.keys()), pred_cands, k)
                jd = jaccardd(list(gold_cands.keys()), pred_cands, k)
                model_name = "run_" + str(index+1)
                report[model_name]['scores']["Jaccard@" + str(k)] = j
                report[model_name]['scores']["Jaccard_dist@" + str(k)] = jd
                # jaccard_reuslts[k][ds2[index]] = j
                # jaccardd_reuslts[k][ds2[index]] = jd
    # print("jaccard reuslts:")
    # for k in jaccard_reuslts:
    #     for index, ds in enumerate(jaccard_reuslts[k]):
    #         print(k, ds, "(" + str(index) + ")",np.mean(jaccard_reuslts[k][ds]))
    # print("jaccard distance reuslts:")
    # for k in jaccardd_reuslts:
    #     for index, ds in enumerate(jaccardd_reuslts[k]):
    #         print(k, ds, "(" + str(index) + ")",np.mean(jaccardd_reuslts[k][ds]))
    # print("-"*80)
    if runjaccard2:
        jaccard_reuslts = {k:{} for k in lst_apk}
        for index1, pred_aux1 in enumerate(preds):
            for index2, pred_aux2 in enumerate(preds):
                for sent in pred_aux1:
                    pred_cands = pred_aux1[sent]
                    gold_cands = pred_aux2[sent]
                    for k in lst_apk:
                        j = jaccard(gold_cands, pred_cands, k)
                        jaccard_reuslts[k][str(index1) + " - " + str(index2)] = j
        print("jaccard reuslts2:")
        for k in jaccard_reuslts:
            for index, ds in enumerate(jaccard_reuslts[k]):
                print(k, ds, np.mean(jaccard_reuslts[k][ds]))
        print("-"*80)
    ###############################################################

    # print(report)
    save_results(file_name+".log", report, p_value=0.05) # see https://amenra.github.io/ranx/metrics/#rank-biased-precision
    # open(file_name+".log","w").write(str(report.to_csv())) 


# print("ES")
# eval(ds1="semeval2012T1/2010_es.csv", ds2="semeval_es_dccuchile_bert-base-spanish-wwm-cased_m1.txt")
# eval(ds1="semeval2012T1/2010_es.csv", ds2="semeval_es_dccuchile_bert-base-spanish-wwm-cased_m2.txt")




# print("EN")
# eval(ds1="semeval2012T1/2012.csv", ds2="semeval_en_bert-large-uncased_m1.txt")
# eval(ds1="semeval2012T1/2012.csv", ds2="semeval_en_bert-large-uncased_m2.txt")

# print("EN")
# eval(ds1="semeval2012T1/2012.csv", ds2="semeval_en_roberta-base_sepFalse.txt")

# eval_lst(file_name = "results_en", ds1="semeval2012T1/2012.csv", ds2=["semeval_en_bert-large-uncased_m1.txt", "semeval_en_bert-large-uncased_m2.txt", "semeval_en_roberta-base_sepFalse.txt"])
# eval_lst(file_name = "results_es", ds1="semeval2012T1/2010_es.csv", ds2=["semeval_es_dccuchile_bert-base-spanish-wwm-cased_m1.txt", "semeval_es_dccuchile_bert-base-spanish-wwm-cased_m2.txt"])

# files = glob.glob("resultats_semeval-QE-doublesent-paraphrases/*_semeval_en_*")
# print("files",files)
# eval_lst(file_name = "results_semeval_en", ds1="git/tsar-shared-task-2022/corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("resultats_semeval-QE-doublesent-paraphrases/*_semeval_es_*")
# print("files",files)
# eval_lst(file_name = "results_semeval_es", ds1="git/tsar-shared-task-2022/corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("resultats_semeval-QE-doublesent-paraphrases/*__tsar_en_*")
# print("files",files)
# eval_lst(file_name = "results_en", ds1="TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_en_trial_gold.tsv", ds2=files, runjaccard2=False)

# ##################################################################################
# ###### Original
# files = glob.glob("qe_output/results_QE/qesemeval_en_*")
# print("files",files)
# eval_lst(file_name = "results_Original_semeval_en_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results/double_semeval_en_*")
# print("files",files)
# eval_lst(file_name = "results_Original_semeval_en_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)


# files = glob.glob("double_sentence/results/double_tsar2022_en_*")
# print("files",files)
# eval_lst(file_name = "results_Original_tsar_en_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("qe_output/results_QE/qetsar2022_en_*")
# print("files",files)
# eval_lst(file_name = "results_Original_tsar_en_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)

# ###### Filtered
# files = glob.glob("qe_output/results_QE_filter/*qesemeval_en_*")
# print("files",files)
# eval_lst(file_name = "___results_Filter_semeval_en_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results_filter/*double_semeval_en_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_semeval_en_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)


# files = glob.glob("double_sentence/results_filter/double_tsar2022_en_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_tsar_en_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("qe_output/results_QE_filter/qetsar2022_en_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_tsar_en_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)

# ###### Filtered
files = glob.glob("qe_output/probs_agree/*qe_semeval_en_*")
print("files",files)
eval_lst(file_name = "results_agree_semeval_en_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)

files = glob.glob("double_sentence/probs_agree/*double_semeval_en_*")
print("files",files)
eval_lst(file_name = "results_agree_semeval_en_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)


files = glob.glob("double_sentence/probs_agree/*double_tsar2022_en_*")
print("files",files)
eval_lst(file_name = "results_agree_tsar_en_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)

files = glob.glob("qe_output/probs_agree/qe_tsar2022_en_*")
print("files",files)
eval_lst(file_name = "results_agree_tsar_en_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)

# ##################################################################################
# ###### Original
# files = glob.glob("qe_output/results_QE/qesemeval_es_*")
# print("files",files)
# eval_lst(file_name = "results_Original_semeval_es_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results/double_semeval_es_*")
# print("files",files)
# eval_lst(file_name = "results_Original_semeval_es_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results/double_tsar2022_es_*")
# print("files",files)
# eval_lst(file_name = "results_Original_tsar_es_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("qe_output/results_QE/qetsar2022_es_*")
# print("files",files)
# eval_lst(file_name = "results_Original_tsar_es_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)

# ###### Filtered
# files = glob.glob("qe_output/results_QE_filter/qesemeval_es_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_semeval_es_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results_filter/double_semeval_es_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_semeval_es_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results_filter/double_tsar2022_es_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_tsar_es_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("qe_output/results_QE_filter/qetsar2022_es_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_tsar_es_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)

###### Filtered
files = glob.glob("qe_output/probs_agree/qe_semeval_es_*")
print("files",files)
eval_lst(file_name = "results_agreesemeval_es_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

files = glob.glob("double_sentence/probs_agree/*double_semeval_es_*")
print("files",files)
eval_lst(file_name = "results_agree_semeval_es_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

files = glob.glob("double_sentence/probs_agree/*double_tsar2022_es_*")
print("files",files)
eval_lst(file_name = "results_agree_tsar_es_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)

files = glob.glob("qe_output/probs_agree/qe_tsar2022_es_*")
print("files",files)
eval_lst(file_name = "results_agree_tsar_es_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)


# ##################################################################################
# ###### Original
# files = glob.glob("qe_output/results_QE/qesemeval_pt_*")
# print("files",files)
# eval_lst(file_name = "results_Original_semeval_pt_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results/double_semeval_pt_*")
# print("files",files)
# eval_lst(file_name = "results_Original_semeval_pt_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results/double_tsar2022_pt_*")
# print("files",files)
# eval_lst(file_name = "results_Original_tsar_pt_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("qe_output/results_QE/qetsar2022_pt_*")
# print("files",files)
# eval_lst(file_name = "results_Original_tsar_pt_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

# ###### Filtered
# files = glob.glob("qe_output/results_QE_filter/qesemeval_pt_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_semeval_pt_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results_filter/double_semeval_pt_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_semeval_pt_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("double_sentence/results_filter/double_tsar2022_pt_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_tsar_pt_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("qe_output/results_QE_filter/qetsar2022_pt_*")
# print("files",files)
# eval_lst(file_name = "results_Filter_tsar_pt_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

###### Filtered
files = glob.glob("qe_output/probs_agree/qe_semeval_pt_*")
print("files",files)
eval_lst(file_name = "results_agree_semeval_pt_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)

files = glob.glob("double_sentence/probs_agree/*double_semeval_pt_*")
print("files",files)
eval_lst(file_name = "results_agree_semeval_pt_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)

files = glob.glob("double_sentence/probs_agree/*double_tsar2022_pt_*")
print("files",files)
eval_lst(file_name = "results_agree_tsar_pt_2sent", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

files = glob.glob("qe_output/probs_agree/qe_tsar2022_pt_*")
print("files",files)
eval_lst(file_name = "results_agree_tsar_pt_qe", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

##################################################################################
# files = ["dictionary/dico_semeval_en.tsv"]
# eval_lst(file_name = "results_Dict_semeval_en", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)
# files = ["dictionary/dico_tsar2022_en_trial.tsv"]
# eval_lst(file_name = "results_Dict_tsar_en", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)

# files = ["dictionary/dico_semeval_es.tsv"]
# eval_lst(file_name = "results_Dict_semeval_es", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)
# files = ["dictionary/dico_tsar2022_es_trial.tsv"]
# eval_lst(file_name = "results_Dict_tsar_es", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)

# files = ["dictionary/dico_semeval_pt.tsv"]
# eval_lst(file_name = "results_Dict_semeval_pt", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)
# files = ["dictionary/dico_tsar2022_pt_trial.tsv"]
# eval_lst(file_name = "results_Dict_tsar_pt", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

##################################################################################



###### Filtered
# files = glob.glob("paraphrase/results_lang_specific/results_filter/tsar2022_*en*")
# print("files",files)
# eval_lst(file_name = "results_Filter_tsar_en_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)
# files = glob.glob("paraphrase/results_lang_specific/results_filter/semeval_*en*")
# print("files",files)
# eval_lst(file_name = "results_Filter_semeval_en_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("paraphrase/results_lang_specific/results_filter/tsar2022_*es*")
# print("files",files)
# eval_lst(file_name = "results_Filter_tsar_es_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("paraphrase/results_lang_specific/results_filter/semeval_*es*")
# print("files",files)
# eval_lst(file_name = "results_Filter_semeval_es_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)


# files = glob.glob("paraphrase/results_lang_specific/results_filter/tsar2022_*pt*")
# print("files",files)
# eval_lst(file_name = "results_Filter_tsar_pt_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("paraphrase/results_lang_specific/results_filter/semeval_*pt*")
# print("files",files)
# eval_lst(file_name = "results_Filter_semeval_pt_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)


# ##################################################################################
# ####  paraphrase: multilang
# files = glob.glob("paraphrase/multilang/results_filter/tsar2022_*en*")
# eval_lst(file_name = "results_Filter_tsar_en_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)
# files = glob.glob("paraphrase/multilang/results_filter/semeval_*en*")
# eval_lst(file_name = "results_Filter_semeval_en_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("paraphrase/multilang/results_filter/tsar2022_*es*")
# eval_lst(file_name = "results_Filter_tsar_es_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("paraphrase/multilang/results_filter/semeval_*es*")
# eval_lst(file_name = "results_Filter_semeval_es_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("paraphrase/multilang/results_filter/tsar2022_*pt*")
# eval_lst(file_name = "results_Filter_tsar_pt_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("paraphrase/multilang/results_filter/semeval_*pt*")
# eval_lst(file_name = "results_Filter_semeval_pt_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)

files = glob.glob("paraphrase/*/probs_agree/_tsar2022_*en*")
eval_lst(file_name = "results_agree_tsar_en_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)
files = glob.glob("paraphrase/*/probs_agree/_semeval_*en*")
eval_lst(file_name = "results_agree_semeval_en_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)

files = glob.glob("paraphrase/*/probs_agree/_tsar2022_*es*")
eval_lst(file_name = "results_agree_tsar_es_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)

files = glob.glob("paraphrase/*/probs_agree/_semeval_*es*")
eval_lst(file_name = "results_agree_semeval_es_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

files = glob.glob("paraphrase/*/probs_agree/_tsar2022_*pt*")
eval_lst(file_name = "results_agree_tsar_pt_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)

files = glob.glob("paraphrase/*/probs_agree/_semeval_*pt*")
eval_lst(file_name = "results_agree_semeval_pt_paraphrase", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)



# ####  paraphrase: monolang.
# files = glob.glob("paraphrase/results_lang_specific/results_filter/tsar2022_*en*")
# eval_lst(file_name = "results_Filter_tsar_en_paraphraseMono", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_en.tsv", ds2=files, runjaccard2=False)
# files = glob.glob("paraphrase/results_lang_specific/results_filter/semeval_*en*")
# eval_lst(file_name = "results_Filter_semeval_en_paraphraseMono", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("paraphrase/results_lang_specific/results_filter/tsar2022_*es*")
# eval_lst(file_name = "results_Filter_tsar_es_paraphraseMono", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_es.tsv", ds2=files, runjaccard2=False)
# files = glob.glob("paraphrase/results_lang_specific/results_filter/semeval_*es*")
# eval_lst(file_name = "results_Filter_semeval_es_paraphraseMono", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_es.tsv", ds2=files, runjaccard2=False)

# files = glob.glob("paraphrase/results_lang_specific/results_filter/tsar2022_*pt*")
# eval_lst(file_name = "results_Filter_tsar_pt_paraphraseMono", ds1="../corpus/cv_tsar_semeval/test_corpus_tsar_pt.tsv", ds2=files, runjaccard2=False)
# files = glob.glob("paraphrase/results_lang_specific/results_filter/semeval_*pt*")
# eval_lst(file_name = "results_Filter_semeval_pt_paraphraseMono", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_pt.tsv", ds2=files, runjaccard2=False)


# # # eval_lst(file_name = "test", ds1="../corpus/cv_tsar_semeval/test_corpus_semeval_en.tsv", ds2=["/mnt/54A453AB3C35D397/CENTAL/shared_task_LexicalSimplification/code_remi/ensembled_generation.txt"], runjaccard2=False)