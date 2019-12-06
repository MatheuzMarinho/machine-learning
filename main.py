import ml

import pandas as pd
import numpy as np


def config(file_name):
    import os
    #Coloca o path no diretorio atual
    path = os.path.dirname(os.path.abspath(__file__))
    
    #Leitura da Planilha
    path_to_file = path+'/datasets/'+file_name
    return path_to_file
    

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


#One classifier for multiple datasets
def main_ml_multiple_csv():
    directory = 'base-djma/'
    files_name = ['CIDADE', 'DESEMBARGADOR','ENUNCIADO','PROCESSO_sample','RELATOR','TIPODEENUNCIADO','TIPODEPROCESSO','VARA']
    output_name = '_DIARIO_MUTIPLE_LABELS'
    
    dfObj = pd.DataFrame(columns=['classificador', 'precisao', 'recall', 'f1', 'acuracia','labels','target_csv'])
    for file_name in files_name:
        print(file_name)
        df = pd.read_csv(config(directory+file_name)+'.csv',encoding ='utf-8')
        X, y = df.conteudo, df.saida 
        #Extrai as LABELS do dataset
        from sklearn.utils.multiclass import unique_labels
        labels = unique_labels(y)

        from sklearn.svm import LinearSVC
        classifier= LinearSVC(random_state=0, tol=1e-5)
        name_clf = 'SVM-LINEAR'
        
        #from sklearn.neighbors import KNeighborsClassifier
        #classifier=KNeighborsClassifier(n_neighbors=5)

        #from sklearn.ensemble import RandomForestClassifier
        #classifier= RandomForestClassifier(n_estimators=10)
        #name_clf = 'RandomForestClassifier'

        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
        name_clf = 'MLPClassifier'
        
        #CrossValidation
        #df = ml.stratifiedShuffleSplitSplitsMultiplus(X,y,classifier,10,name_clf,labels,file_name)
        
        #TrainTest 75 treino 25 teste
        df = ml.trainTesteMultipleDataset(X,y,classifier,30,name_clf,labels,file_name)
        
        dfObj = dfObj.append(df,ignore_index=True)   
    dfObj.to_csv(ml.config(name_clf+output_name)+'.csv',index=False)

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

def main_ml_multiple_classifiers(file_name):
    df = pd.read_csv(config(file_name)+'.csv',encoding ='utf-8')
    df['class'] = np.where(df['class'] == 'F', 'F','RNF')
    X, y = df.requirement_text, df.iloc[:,-1]
    output_name = 'NFR_MULTIPLE_CLASSIFIERS'
    list_classifiers = []

    from sklearn.ensemble import RandomForestClassifier
    classifier= RandomForestClassifier(n_estimators=10)
    name_clf = 'RandomForestClassifier'
    list_classifiers.append([classifier,name_clf])
    
    from sklearn.svm import LinearSVC
    classifier= LinearSVC(random_state=0, tol=1e-5)
    name_clf = 'LinearSVC'
    list_classifiers.append([classifier,name_clf])

    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
    name_clf = 'MLPClassifier'
    #list_classifiers.append([classifier,name_clf])
    
    from sklearn.utils.multiclass import unique_labels
    labels = unique_labels(y)

    #dfObj = ml.trainTesteMultipleClassifiers(X,y,list_classifiers,30,labels,file_name)
    dfObj = ml.stratifiedShuffleSplitSplitsMultiplusClassifiers(X,y,list_classifiers,10,labels,file_name)
    dfObj.to_csv(ml.config(output_name)+'.csv',index=False)

if __name__ == "__main__":
    #define o nome do arquivo csv a ser lido.
    file_name = 'nfr_dataset' 
    #main_ml(file_name)
    #main_ml_multiple_csv()
    main_ml_multiple_classifiers(file_name)
    



