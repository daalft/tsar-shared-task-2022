#!/usr/bin/env python
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoModel, AutoModelWithLMHead, AutoTokenizer
import logging
import re
logging.basicConfig(level=logging.INFO)# OPTIONAL

tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
model = BertForMaskedLM.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
model.eval()
# model.to('cuda')  # if you have gpu

def paraphrase(text, num_return_sequences, num_beams, max_length=128):
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
  generated_ids = model.generate(input_ids=input_ids, num_return_sequences=num_return_sequences, num_beams=num_beams, max_length=max_length, no_repeat_ngram_size=2, repetition_penalty=3.5, length_penalty=1.0, early_stopping=True)
  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
  return preds

def predict_masked_sent(text, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    print(" ".join(tokenized_text))
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
        predictions.append(predicted_token)
        token_weight = top_k_weights[i]
        masks.write("[MASK]:\t"+predicted_token+"\t weights:"+str(float(token_weight))+"\n")
    return predictions



#from transformers import PegasusForConditionalGeneration, PegasusTokenizer
#model_name = 'mrm8488/bert2bert_shared-spanish-finetuned-paus-x-paraphrasing'
model_name = 'seduerr/mt5-paraphrases-espanol'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
ptokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
pmodel = AutoModel.from_pretrained(model_name)

#num_beams = 10
#num_return_sequences = 10

for nb in [10,20]:
    num_beams = nb
    for nrs in [5,10]:
        if nrs >= nb:
            num_return_sequences = nrs
            f = open("tsar2022_es_trial_none.tsv",'r')
            out = open("resultats_es/"+str(num_beams)+"_"+str(num_return_sequences)+".tsv","w")
            masks = open("masks_es/"+str(num_beams)+"_"+str(num_return_sequences)+".txt",'w')
            for l in f:
                sent,word = l.rstrip().split("\t")
                srep = sent.replace(word,"[MASK]")
                #print(srep)
                context = sent
                masks.write(l)
                paraphrases = paraphrase(context,num_return_sequences,num_beams)
                for p in paraphrases:
                    #print(context+" "+str(len(tokenizer.tokenize(context))+len(tokenizer.tokenize(p))+len(tokenizer.tokenize(srep))))
                    if len(tokenizer.tokenize(context))+len(tokenizer.tokenize(p))+len(tokenizer.tokenize(srep)) <= 512:
                        context += " "+p
                    else:
                        break
                context += " "+srep
                #print(context)
                pred = predict_masked_sent(context, top_k=20)
                predfilt = []
                for p in pred:
                    x = re.search("^[a-zA-Z].*[a-zA-Z]$",p)
                    if word not in p and x != None:
                        predfilt.append(p)
                predok = []
                for i in range(10):
                    predok.append(predfilt[i])
                final = sent+"\t"+word+"\t"+"\t".join(predok)+"\n"
                #print(final)
                out.write(final)
            f.close()
            out.close()
            masks.close()
