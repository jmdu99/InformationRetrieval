import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
import re
import nltk
from nltk import PorterStemmer, WordNetLemmatizer
import string as st
import numpy as np
import os

'''
Helper script to transform a LOINC dataset (excel format) into svmlight files. 

From the generated files, an AdaRank model can be trained using 
Ruey-Cheng Chen library: https://github.com/rueycheng/AdaRank
'''

def load_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
		
def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))

def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]
	
def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]

def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]

def stemming(text):
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]

def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]

def return_sentences(tokens):
    return " ".join([word for word in tokens])

def preprocess_text(text):
    return return_sentences(
    lemmatize(stemming (remove_stopwords(remove_small_words(
                   (tokenize(remove_punct(text))))))))

def excel_to_df(excel_file):
    sheets_dict = pd.read_excel(excel_file, sheet_name=None, skiprows=1, header=1)
    all_sheets = []
    for name, sheet in sheets_dict.items():
        all_sheets.append(sheet)
    df= pd.concat(all_sheets)
    df.reset_index(inplace=True, drop=True)
    return df

def get_train_test_df(df):
    # Shuffle dataframe
    df = df.sample(frac=1, random_state = 2)
    train_size = int(0.7 * len(df))
    train_set = df[:train_size].sort_values(by=['qid'])
    test_set = df[train_size:].sort_values(by=['qid'])
    return train_set, test_set

def add_docnos(docnos, file, train_docnos, train_file, test_docnos, test_file):
    if file == 'loinc_dataset-v2_without_docnos.dat':
        out_file = 'loinc_dataset-v2.dat'
        out_train_file = 'train_loinc_dataset-v2.dat'
        out_test_file = 'test_loinc_dataset-v2.dat'
    else:
        out_file = 'extended_loinc_dataset-v2.dat'
        out_train_file = 'extended_train_loinc_dataset-v2.dat'
        out_test_file = 'extended_test_loinc_dataset-v2.dat'
        
    in_files = [file, train_file, test_file]
    out_files = [out_file, out_train_file, out_test_file]
    
    for i in range(3):
        if i == 0:
            data = docnos
        elif i == 1:
            data = train_docnos
        else:
            data = test_docnos
        
        with open(in_files[i]) as fin, open(out_files[i], 'w') as fout:
                index = 0
                for line in fin:
                    fout.write(line.replace('\n', ' ' + str(data[index]) + '\n'))
                    index += 1
    return out_file, out_train_file, out_test_file
    
def df_to_svmlight_files(df):
    if len(df) == 201:
        file = 'loinc_dataset-v2_without_docnos.dat'
        train_file = 'train_loinc_dataset-v2_without_docnos.dat'
        test_file = 'test_loinc_dataset-v2_without_docnos.dat'
    else:
        file = 'extended_loinc_dataset-v2_without_docnos.dat'
        train_file = 'extended_train_loinc_dataset-v2_without_docnos.dat'
        test_file = 'extended_test_loinc_dataset-v2_without_docnos.dat'
    
    # Download nltk stopwords and wordnet if they are not already installed
    load_nltk_data()
    
    # Get document features
    tfidf = TfidfVectorizer()
    df['clean_text'] = df['long_common_name'].apply(lambda x: preprocess_text(x))
    X = tfidf.fit_transform(df['clean_text']).toarray()
    
    # Each document in the df has a set of features
    df['features'] = X.tolist()
    # Split dataset in training (70%) and testing (30%)
    train_df, test_df = get_train_test_df(df)
    
    # DataFrame to numpy array
    qid = df['qid'].to_numpy()
    y = df['relevance'].to_numpy()
    X_train = np.array(train_df['features'].values.tolist())
    qid_train = train_df['qid'].to_numpy()
    y_train = train_df['relevance'].to_numpy()
    X_test = np.array(test_df['features'].values.tolist())
    qid_test = test_df['qid'].to_numpy()
    y_test = test_df['relevance'].to_numpy()
  
    # Numpy arrays into svmlight files
    dump_svmlight_file(X, y, file, query_id=qid)
    dump_svmlight_file(X_train, y_train, train_file, query_id=qid_train)
    dump_svmlight_file(X_test, y_test, test_file, query_id=qid_test)
    
    # Add docnos to svmlight files
    out_file, out_train_file, out_test_file = add_docnos(df['docno'].tolist(), file, train_df['docno'].tolist()
                                                         , train_file, test_df['docno'].tolist(), test_file)
    
    # Remove files without docnos
    my_dir = os.getcwd()
    for fname in os.listdir(my_dir):
        if 'docnos' in fname :
            os.remove(os.path.join(my_dir, fname))
    return out_file, out_train_file, out_test_file
