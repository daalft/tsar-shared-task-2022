#!/usr/bin/env python

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
import re
logging.basicConfig(level=logging.INFO)# OPTIONAL



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
# model.to('cuda')  # if you have gpu


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



from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
ptokenizer = PegasusTokenizer.from_pretrained(model_name)
pmodel = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams):
  batch = ptokenizer([input_text],truncation=False,padding='longest',max_length=1000, return_tensors="pt").to(torch_device)
  translated = pmodel.generate(**batch,max_length=1000,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=30)
  tgt_text = ptokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

#num_beams = 10
#num_return_sequences = 10

for nb in range(10,20):
    num_beams = nb
    for nrs in range(1,20):
        if nrs >= nb:
            num_return_sequences = nrs
            f = open("tsar2022_en_trial_none.tsv",'r')
            out = open("resultats/"+str(num_beams)+"_"+str(num_return_sequences)+".tsv","w")
            masks = open("masks/"+str(num_beams)+"_"+str(num_return_sequences)+".txt",'w')
            for l in f:
                sent,word = l.rstrip().split("\t")
                srep = sent.replace(word,"[MASK]")
                context = sent
                masks.write(l)
                paraphrases = get_response(context,num_return_sequences,num_beams)
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
