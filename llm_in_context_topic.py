import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import evaluate
from transformers import TrainingArguments
from datasets import Dataset, DatasetDict
from transformers import Trainer

from transformers import OPTForCausalLM
from peft import LoraConfig
import timeit  # calcular metrica de tempo
from transformers import set_seed

import wandb

def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', max_length=256, truncation=True)

# reprodutibilidade da solucao
import random, torch, numpy
SEED=42
random.seed(SEED); torch.manual_seed(SEED); numpy.random.seed(seed=SEED) # reproducibily soluction  
dataset='dblp' 
nome_llm='llama3' # flan # bloomz # llama13b
limiar = 2 # 2 para enviar tudo para classificacao da LLM 

#topic notebook anterior


if nome_llm == "llama3":
    nome_modelo = "meta-llama/Meta-Llama-3.1-8B-Instruct"
   
    model = AutoModelForCausalLM.from_pretrained(nome_modelo, 
            token=TOKEN_USUARIO, torch_dtype="auto", device_map="auto", load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo, token=TOKEN_USUARIO)


import timeit  # calcular metrica de tempo



mapeamento = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4 : 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10 : 'K'} 

imprimir=""
for index_fold in [0]:
    ini = timeit.default_timer()    

    #leitura--
    df = pd.read_parquet(f'kaggle/input/datasets-sentiment/{dataset}.parquet')
    folds = pd.read_parquet(f'kaggle/input/datasets-sentiment/{dataset}_folds.parquet')
    representacao = pd.read_pickle(f'kaggle/input/datasets-save-model-representation/{dataset}_roberta_representacao_fold_{index_fold}.pickle')    
    roberta_metricas = pd.read_pickle(f'kaggle/input/datasets-save-model-representation/{dataset}_roberta_metricas_fold_{index_fold}.pickle')
    
    test_roberta = representacao.iloc[   folds['test_idxs'][index_fold] ] 
    df['roberta'] = representacao['roberta']
    
    #menos confiantes----
    df_test = df.loc[folds.iloc[index_fold]['test_idxs']]
    df_test['prob_roberta'] = roberta_metricas.iloc[0]['y_prob']
    df_test['pred_roberta'] = roberta_metricas.iloc[0]['y_pred']
    confianca=[]
    for index_doc in range(len(df_test)):
        confianca.append( np.max(roberta_metricas.iloc[0]['y_prob'][index_doc]))
    df_test['confianca'] = confianca
    df_baixa_confianca = df_test[df_test['confianca']<limiar]
    print(len(df_baixa_confianca))
    
    #menos confiantes----
    
    #prompt----
    train_roberta = representacao.iloc[   folds['train_idxs'][index_fold] ] 
    
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=2, metric='cosine') # 1 para zero-shot
    neigh.fit(list(train_roberta['roberta']))
    
    results = []
    df_baixa_confianca = df_test # classifica tudo por LLM
    for i_menos_confiantes in range(len(df_baixa_confianca)):   
        if i_menos_confiantes % 10 == 0:
            print(i_menos_confiantes)
        
        prompt = f"Classify the topic of the text exclusively among the references:\n"        
        posicoes_train_similar = neigh.kneighbors([df_baixa_confianca.iloc[i_menos_confiantes]['roberta']],return_distance=False)[0] #posicao no treino        
        for posicao_train_similar in posicoes_train_similar: # monta o prompt com os exemplos    
            doc_similar = df.iloc[train_roberta.iloc[posicao_train_similar]['idx']] # idx para pegar o indice real do texto                    

            prompt += f"Input: {doc_similar['text']}.\nReference:\nA. Computer vision\nB. Computational linguistics\nC. Biomedical engineering\nD. Software engineering\nE. Graphics\nF. Theory of Computation\nG. Security and cryptography\nH. Signal processing\nI. Robotics\nJ. Computer Applications\nK. Theory\nAnswer: {mapeamento[doc_similar['label']]}\n"            


        prompt += f"Input: {df_baixa_confianca.iloc[i_menos_confiantes]['text']}.\nReference:\nA. Computer vision\nB. Computational linguistics\nC. Biomedical engineering\nD. Software engineering\nE. Graphics\nF. Theory of Computation\nG. Security and cryptography\nH. Signal processing\nI. Robotics\nJ. Computer Applications\nK. Theory\nAnswer:"



        inputs = tokenizer.encode(prompt, max_length=1024, return_tensors="pt").to("cuda")                                 
        set_seed(42)
        outputs = model.generate(inputs, max_new_tokens=2)
        results.append(tokenizer.decode(outputs[0][len(inputs[0]):]))
        
    
    df_baixa_confianca[nome_modelo] = results

    
    # converter resultados----
    pred = []
    for i in range(len(df_baixa_confianca)):
        entrou=False
        for k, v in mapeamento.items():
        #if 'B' in df_baixa_confianca.iloc[i][nome_modelo]:
            if v == df_baixa_confianca.iloc[i][nome_modelo].replace('\n', '').strip():        
                pred.append(k)
                entrou=True
                break
        if entrou==False: #atribui maioria        
            pred.append(int(df['label'].value_counts().idxmax()))
    
    df_baixa_confianca[nome_modelo +'_pred'] = pred
    df_test.loc[list(df_baixa_confianca.idx), 'pred_roberta'] = list(df_baixa_confianca[nome_modelo +'_pred'])    
    #df_test[nome_modelo]=np.nan
    df_test.loc[list(df_baixa_confianca.idx), nome_modelo] = list(df_baixa_confianca[nome_modelo])    
    #metricas----    
    from sklearn.metrics import f1_score
    micro = f1_score(list(df_test['label']), list(df_test['pred_roberta']), average='micro')
    macro = f1_score(list(df_test['label']), list(df_test['pred_roberta']), average='macro')
    imprimir += f"{dataset}_{nome_modelo}\t{index_fold}\t{micro}\t{macro}\t{timeit.default_timer() - ini}\t{list(df_test['pred_roberta'])}\n"
    print(imprimir) 
    df_test.to_parquet(f"{dataset}_{index_fold}.parquet")
