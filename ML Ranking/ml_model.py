import helper
from adarank_lib.metrics import NDCGScorer
from adarank_lib.adarank import AdaRank
from sklearn.datasets import load_svmlight_file
import os
from adarank_lib.utils import load_docno, print_ranking

'''
This script uses the Ruey-Cheng Chen library to train and test an AdaRank model:
https://github.com/rueycheng/AdaRank

The model is trained with a training svmlight file and 
evaluated with a test svmlight file. 
'''

# Excel files
excel_doc = 'data/labelled_loinc_dataset-v2.xlsx'
extended_excel_doc = 'data/extended_labelled_loinc_dataset-v2.xlsx'

# Dataframes
df = helper.excel_to_df(excel_doc)
extended_df = helper.excel_to_df(extended_excel_doc)

# Svmlight files
os.chdir('data/svmlight_files')
file, train_file, test_file  = helper.df_to_svmlight_files(df)
extended_file, extended_train_file, extended_test_file = helper.df_to_svmlight_files(extended_df)

def get_adarank_score_and_ranking(excel_doc):
    if excel_doc == extended_excel_doc:
        tr_file = extended_train_file
        tst_file = extended_test_file
        ranking_file = 'extended_ranking.txt'
    else:
        tr_file = train_file
        tst_file = test_file
        ranking_file = 'ranking.txt'
   
    X_train, y_train, qid_train = load_svmlight_file(tr_file, query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(tst_file, query_id=True)
    
    '''
    Normalized Discounted Cumulative Gain scorer:
        A measure of ranking quality that is often used to measure effectiveness 
        of web search engine algorithms or related applications.
    '''
    
    '''
    Run AdaRank for 100 iterations optimizing for NDCG@10. 
    When no improvement is made within the previous 10 iterations, 
    the algorithm will stop.
    '''
    
    model = AdaRank(max_iter=100, estop=10, scorer=NDCGScorer(k=10)).fit(X_train, y_train, qid_train)
    pred = model.predict(X_test, qid_test)
    
    # nDCG scores
    for k in (1, 2, 3, 4, 5, 10, 20):
        score = NDCGScorer(k=k)(y_test, pred, qid_test).mean()
        print('nDCG@{}\t{}'.format(k, score))
    
    # Return ranking
    docno = load_docno(tst_file, letor=False)
    os.chdir('../rankings')
    print_ranking(qid_test, docno, pred, output=open(ranking_file, 'w'))
    os.chdir('../svmlight_files')
    
# Get scores for excel documents
print('nDCG scores for original dataset:')
get_adarank_score_and_ranking(excel_doc)

'''
nDCG scores for original dataset:
nDCG@1	0.3333333333333333
nDCG@2	0.5436432511904858
nDCG@3	0.5436432511904858
nDCG@4	0.5436432511904858
nDCG@5	0.5436432511904858
nDCG@10	0.5436432511904858
nDCG@20	0.5436432511904858
'''
print('\nnDCG scores for extended dataset:')
get_adarank_score_and_ranking(extended_excel_doc)

'''
nDCG scores for extended dataset:
nDCG@1	0.2
nDCG@2	0.2
nDCG@3	0.2
nDCG@4	0.24042146930010952
nDCG@5	0.3329185548465836
nDCG@10	0.38359650613556295
nDCG@20	0.4715501865835844
'''
