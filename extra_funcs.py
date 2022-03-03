#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:11:54 2022

@author: alexisguillot
"""
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from pathlib import Path 
from bs4 import BeautifulSoup as bs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from PIL import Image
import torch
from transformers import BertTokenizer, BertModel
from DeepLearningMethod import SimpleNeuralNet



def num_apperances_of_tag(tag_name, html):
    
    return len(html.find_all(tag_name))


def train(data, random_state, model, eval_every, hidden_size_layer, learning_rate, epochs, split_ratio):
    idx_to_class = {
    0: " social",
    1: " article",
    2: " landing",
    3: " commercial",
    4: " company_information"
    }
    y_pred = None
    classifier = None
    y = data[' label']
    X = data.drop(columns=[' label', 'hash'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    
    if model == 'NN':
        
        X_train_torch = torch.tensor(X_train.values, dtype=torch.float)
        y_train_numeric, vals = pd.factorize(y_train)
        y_train_torch = torch.tensor(y_train_numeric)
        
        X_test_torch = torch.tensor(X_test.values, dtype=torch.float)
        y_test_numeric, vals = pd.factorize(y_test)
        y_test_torch = torch.tensor(y_test_numeric)
        
       
        model = SimpleNeuralNet(X_train_torch.shape[1], hidden_size_layer, len(vals))
        
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr =learning_rate)
        
        
        for epoch in range(epochs):
            model.zero_grad()
            output = model(X_train_torch)
            loss =criterion(output,y_train_torch)
            loss.backward()
            # print(loss)
            optimizer.step()
            if epoch%eval_every==0:
                model.eval()
                y_pred_torch = model(X_test_torch)
                preds = torch.argmax(y_pred_torch, dim=1)
                corrects = (preds==y_test_torch).float().sum()
                print(corrects/X_test_torch.shape[0])
            
        y_pred_torch = torch.argmax(model(X_test_torch), dim=1).detach().numpy()
        y_pred = np.vectorize(idx_to_class.get)(y_pred_torch)
        
    if model == 'SVM':
        classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    else:
        classifier = RandomForestClassifier(random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
    return y_pred, y_test       
        

def evaluate(y_pred, y_test):
    print(classification_report(y_test, y_pred))

def load_data(num_samples = -1):
    path_data = Path().absolute() / "data"
    df = pd.read_csv(path_data / "labels.csv")
    shuffled_df = df.sample(frac=1, random_state = 0)
    small_df = df
    if num_samples != -1:
        
        small_df = shuffled_df.iloc[:num_samples]
    
    
    list_n_a = []
   
    list_n_links = []
    list_n_buttons = []
    list_n_images = []
    list_n_h1 = []
    list_n_h2 = []
    list_n_h3 = []
    list_n_li = []
    list_n_ul = []
    
    list_n_currencies = []
    list_n_digits = []
    

    list_embeddings = []

    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained("bert-base-multilingual-cased")

    
    
    
        
    for h in small_df["hash"]:
        
        with open(path_data / "pages" / f"{h}.html") as fp:
            soup = bs(fp, 'html.parser')
            
            maincorpus = ""
            
            if soup.findAll('p'):
                
                for paragraph in soup.findAll('p'):
                    maincorpus+=paragraph.get_text()
            
            
            count_currencies = 0 + maincorpus.count('€')+maincorpus.count('£')+maincorpus.count('$')
            n_digits = sum(c.isdigit() for c in maincorpus)
            try:
                
                title = soup.find('title').get_text()
            except:
                title = 'none'
            encoded_input = tokenizer(title, return_tensors='pt')
            output = model(**encoded_input)
            title_tokenized = output[1].flatten().detach().numpy()
                        
            list_n_a.append(num_apperances_of_tag("a", soup))
            list_n_links.append(num_apperances_of_tag("link", soup))
            list_n_buttons.append(num_apperances_of_tag("button", soup))
            list_n_images.append(num_apperances_of_tag("img", soup))
            list_n_h1.append(num_apperances_of_tag("h1", soup))
            list_n_h2.append(num_apperances_of_tag("h2", soup))
            list_n_h3.append(num_apperances_of_tag("h3", soup))
            list_n_li.append(num_apperances_of_tag("li", soup))
            list_n_ul.append(num_apperances_of_tag("ul", soup))
            list_n_currencies.append(count_currencies)
            list_n_digits.append(n_digits)
            list_embeddings.append(title_tokenized)

    
    small_df['n_a'] = pd.Series(list_n_a).values
    small_df['n_links'] = pd.Series(list_n_links).values
    small_df['n_buttons'] = pd.Series(list_n_buttons).values
    small_df['n_images'] = pd.Series(list_n_images).values
    small_df['n_h1'] = pd.Series(list_n_h1).values
    small_df['n_h2'] = pd.Series(list_n_h2).values
    small_df['n_h3'] = pd.Series(list_n_h3).values
    small_df['n_li'] = pd.Series(list_n_li).values
    small_df['n_ul'] = pd.Series(list_n_ul).values
    small_df['n_currencies'] = pd.Series(list_n_currencies).values
    small_df['n_digits'] = pd.Series(list_n_digits).values
    small_df.reset_index(inplace=True)
    small_df = pd.concat([small_df, pd.DataFrame(np.array(list_embeddings))], axis =1 )
    
    return small_df