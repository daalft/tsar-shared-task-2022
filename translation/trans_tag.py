import pandas as pd
import os
from google.cloud import translate_v2
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= r"/home/isabelle/Documents/PROJECTS/Human_Remains/translationAPI/googlecloudkey_hr-translation.json"
translate_client=translate_v2.Client()

file="/home/isabelle/Downloads/2012.csv"

column_names=['sentence', 'cand_1', 'cand_2', 'cand_3', 'cand_4', 'cand_5', 'cand_6', 'cand_7', 'cand_8', 'cand_9']
df=pd.read_csv(file, sep='\t', names=column_names, engine='python', quoting=3)

def rem_header(sentence):
    return sentence.replace('<head>', '')

def translation_tokens(list_tokens):
    text=', '.join(list_tokens)
    #target='es'
    target='pt'
    output = translate_client.translate(text,target_language=target)
    result=output['translatedText']
    return result

def translation_sents(sentence_clean):
    text=sentence_clean
    #target='es'
    target='pt'
    output = translate_client.translate(text,target_language=target)
    result=output['translatedText']
    return result

def clean_list(list_tokens):
    list_tokens = [token for token in list_tokens if token != None]
    return list_tokens

def spread(string):
    l=string.split(", ")
    return l


df['sentence_clean']=df['sentence'].apply(rem_header)
#translate all candidates together in case it produces more varied translations
df['list_tokens'] = df[['cand_1', 'cand_2', 'cand_3', 'cand_4', 'cand_5', 'cand_6', 'cand_7', 'cand_8', 'cand_9']].values.tolist()
df['list_tokens']= df['list_tokens'].apply(clean_list)

df['trans_tokens']= df['list_tokens'].apply(translation_tokens)
df['trans_sent']=df['sentence_clean'].apply(translation_sents)

#translated string back to list to recreate df
df['trans_list']=df['trans_tokens'].apply(spread)
split_df=pd.DataFrame(df["trans_list"].tolist(), columns=['cand_1', 'cand_2', 'cand_3', 'cand_4', 'cand_5', 'cand_6', 'cand_7', 'cand_8', 'cand_9'])
pt_df = pd.concat([df['trans_sent'], split_df, df['trans_list']], axis=1)



import spacy
nlp = spacy.load("pt_core_news_sm")
#nlp = spacy.load("es_core_news_sm")

def add_head(vec):
    translation = vec[0]
    list_tokens = vec[1]
    sent=nlp(translation)
    new_sent=''
    for token in sent:
        if token.lemma_ in list_tokens:
            new_sent+='<head>'+token.text+'<head> '
        else:
            new_sent+=token.text+' '
    return new_sent


pt_df['annotation']=pt_df[['trans_sent', 'trans_list']].apply(add_head, axis=1)
pt_df_a=pd.concat([pt_df['annotation'], split_df], axis=1)
pt_df_a.to_csv('translation_pt_tag.csv')
