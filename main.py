import ml
import evaluate

import pandas as pd
import numpy as np


def config(file_name):
    import os
    #Coloca o path no diretorio atual
    path = os.path.dirname(os.path.abspath(__file__))
    
    #Leitura da Planilha
    path_to_file = path+'/datasets/'+file_name
    return path_to_file
    
def evaluate_csv(name_file):
    import os
    #Coloca o path no diretorio atual
    path = os.path.dirname(os.path.abspath(__file__))
    directory_output = 'output'
    path_final = path+'/'+directory_output+'/'
    evaluate.evaluate_csv(path_final+name_file)

def ml_analysis(X,y):
    #Extrai as LABELS do dataset
    from sklearn.utils.multiclass import unique_labels
    labels = unique_labels(y)
    
    #Definição do Classificador
    from sklearn import svm
    classifier = svm.SVC(gamma='scale')
    
    name_clf = 'SVM'

    #from sklearn.naive_bayes import MultinomialNB
    #classifier = MultinomialNB()

    #Chamadas
    #ml.single_round(classifier,X,y)
    ml.stratifiedShuffleSplitSplits(X,y,classifier,10,name_clf,labels)
    #ml.trainTeste(X,y,classifier,10,name_clf,labels)

def main_ml(file_name):
  
    df = pd.read_csv(config(file_name)+'.csv',encoding ='utf-8')

    #ml.overview(df)

    #Código indivual por problema...

    #Change Columns - F and NF
    df['class'] = np.where(df['class'] == 'F', 'F','RNF')

    #SELECT ONLY WITH VALUES
    #df = df.loc[df['class'].isin(['US','SE','O','PE'])]
    #df= df.reset_index()

    #Separacao das features e dos targets (X,Y)
    X, y = df.requirement_text, df.iloc[:,-1]
    #print(y)



    ml_analysis(X,y)


if __name__ == "__main__":
    #define o nome do arquivo csv a ser lido.
    
    #file_name = 'nfr_dataset' 
    #main_ml(file_name)
    
    file_name = 'SVM_D27-11-2019T11_51_11'
    evaluate_csv(file_name)




