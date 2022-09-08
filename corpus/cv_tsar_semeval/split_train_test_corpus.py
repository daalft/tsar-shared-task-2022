from collections import Counter
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
def load_corpus(path, sep = "\t", use_simplest_only=False):
	corpus = []
	with open(path) as input_file:
		for ln in input_file:
			cols = ln.strip().split(sep)
			sent = cols[0]
			if "<head>" not in sent or len(sent.split("<head>"))>3:
				continue
			_, w, _ = sent.split("<head>")
			# ignore the orignal word and all the following
			cands = []
			for c in cols[1:]:
				if len(c.strip())==0:
					continue
				if use_simplest_only and c == w:
					break
				cands.append(c)
			if len(cands) > 1:
				corpus.append([sent, cands])
	return corpus

def load_tsar(path, sep = "\t", use_simplest_only=False):
	corpus = []
	with open(path) as input_file:
		for ln in input_file:
			cols = ln.strip().split(sep)
			sent = cols[0]
			w = cols[1]
			sent = sent.replace(w, "<head>"+w+"<head>")
			
			cands = []
			for c in cols[2:]:
				if len(c.strip())==0:
					continue
				if use_simplest_only and c == w: # ignore the orignal word and all the following
					break
				cands.append(c)
			cnt_cands = Counter(cands)
			if len(cands) > 1:
				corpus.append([sent, [w for w, c in cnt_cands.most_common()], cnt_cands])
	return corpus

def split_corpus(corpus, corpus_name, sep = "\t", perctest = 80):
	# corpus = load_corpus(corpus_path, use_simplest_only=use_simplest_only)
	corpus_split = int(round( (len(corpus)/100)* perctest,0))
	train_corpus = corpus[:corpus_split]
	test_corpus = corpus[corpus_split:]
	print("corpus",len(corpus), "train_corpus", len(train_corpus), "test_corpus", len(test_corpus))


	with open("train_corpus_"+corpus_name+".tsv","w") as output_file:
		for aux in train_corpus:
			sent, cands = aux[0], aux[1]
			if "<head>" not in sent or len(sent.split("<head>"))>3:
				continue
			output_file.write(sent +sep+ sent.split("<head>")[1] +sep+ sep.join(cands) + "\n")

	with open("test_corpus_"+corpus_name+".tsv","w") as output_file:
		for aux in test_corpus:
			sent, cands = aux[0], aux[1]
			if "<head>" not in sent or len(sent.split("<head>"))>3:
				continue
			output_file.write(sent +sep+ sent.split("<head>")[1] +sep+ sep.join(cands) + "\n")

def __get_pairs(corpus):
	pairs = []
	for sent, _, cnt in corpus:
		lst_level_freq = {}
		freqs = set()
		for w in cnt:
			if cnt[w] not in lst_level_freq:
				lst_level_freq[cnt[w]] = []
			lst_level_freq[cnt[w]].append(w)
			freqs.add(cnt[w])
		freqs = sorted(freqs, reverse=True)
		for i1, f1 in enumerate(freqs):
			for f2 in freqs[i1+1:]:
				for w1 in lst_level_freq[f1]:
					for w2 in lst_level_freq[f2]:
						pairs.append( [w1, w2, 0, sent] )
				# pairs.append( [f2, f1, 1] )
	return pairs


def split_corpus_tsar(corpus, corpus_name, sep = "\t", perctest = 80):
	

	corpus_split = int(round( (len(corpus)/100)* perctest,0))
	train_corpus = corpus[:corpus_split]
	test_corpus = corpus[corpus_split:]
	pairs = __get_pairs(train_corpus)
	pairs_hard_test = __get_pairs(test_corpus)
	# print("pairs",pairs[:5])
	# print("pairs_hard_test",pairs_hard_test[:5])
	# print("-"*80)
	# print("-"*80)
	# print("-"*80)
	# print("train_corpus",train_corpus[:2])
	# print("*"*80)
	# print("*"*80)
	# print("*"*80)
	# print("test_corpus",test_corpus[:2])

	# train_pairs, test_pairs = train_test_split(pairs, test_size=1-(perctest/100), shuffle=True)	
	kf = RepeatedKFold(n_splits=4, n_repeats=3)
	print("len total",len(pairs))
	pairs = np.asarray(pairs)
	for i, (train, test) in enumerate(kf.split(pairs)):
		with open("train_corpus_"+corpus_name+"_fold" +str(i) + ".tsv","w") as output_file:
			for w1, w2, index, sent in pairs[train]:
				index = int(index)
				# print(type(w1), type(w2), type(index), type(sent))
				output_file.write(w1 +"\t"+ w2 +"\t"+ str(index) +"\t"+ sent+ "\n")
				output_file.write(w2 +"\t"+ w1 +"\t"+ str(index+1) +"\t"+ sent+ "\n")
		with open("test_corpus_"+corpus_name+"_fold" +str(i) + ".tsv","w") as output_file:
			for w1, w2, index, sent in pairs[test]:
				index = int(index)
				output_file.write(w1 +"\t"+ w2 +"\t"+ str(index) +"\t"+ sent+ "\n")
				output_file.write(w2 +"\t"+ w1 +"\t"+ str(index+1) +"\t"+ sent+ "\n")
	with open("hard_test_corpus_"+corpus_name+"_fold" +str(i) + ".tsv","w") as output_file:
		for w1, w2, index, sent in pairs_hard_test:
			index = int(index)
			output_file.write(w1 +"\t"+ w2 +"\t"+ str(index) +"\t"+ sent+ "\n")
			output_file.write(w2 +"\t"+ w1 +"\t"+ str(index+1) +"\t"+ sent+ "\n")


split_corpus(corpus = load_corpus("semeval2012T1/2010_es.csv"), corpus_name = "semeval_es", sep = "\t", perctest = 80)
split_corpus(corpus = load_corpus("semeval2012T1/2012.csv"), corpus_name = "semeval_en", sep = "\t", perctest = 80)
split_corpus(corpus = load_tsar("TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_en_trial_gold.tsv"), corpus_name = "tsar_en", sep = "\t", perctest = 70)
split_corpus(corpus = load_tsar("TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_es_trial_gold.tsv"), corpus_name = "tsar_es", sep = "\t", perctest = 70)
split_corpus(corpus = load_tsar("TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_pt_trial_gold.tsv"), corpus_name = "tsar_pt", sep = "\t", perctest = 70)

split_corpus_tsar(corpus = load_tsar("TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_pt_trial_gold.tsv"), corpus_name = "tsar_pt", sep = "\t", perctest = 70)
split_corpus_tsar(corpus = load_tsar("TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_en_trial_gold.tsv"), corpus_name = "tsar_en", sep = "\t", perctest = 70)
split_corpus_tsar(corpus = load_tsar("TSAR-2022-Shared-Task-main/datasets/trial/tsar2022_es_trial_gold.tsv"), corpus_name = "tsar_es", sep = "\t", perctest = 70)