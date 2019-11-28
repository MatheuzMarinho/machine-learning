import pandas as pd
import numpy as np

#LIBS PARA PLOT
import seaborn as sns
import matplotlib.pyplot as plt



def overview(df):
    #print(df.shape)
    #print(df.head)

    #GERAL
    print(df['class'].value_counts())
    # Com Seaborn
    ax = sns.catplot('class', data=df, kind='count', order = df['class'].value_counts().index)
    ax.set(xlabel='Quantidade', ylabel='Classe', title='Visão Geral')
    plt.show()


def createDataFrameMetrics(precisao, recall, f1, acuracia, name_clf,labels, target):
    df = pd.DataFrame(
        {'classificador': name_clf,
        'precisao': [precisao],
        'recall': [recall],
        'f1': [f1],
        'acuracia': [acuracia],
        'labels':[labels.tolist()],
        'target_csv':target
        }
    )
    return df
    
def saveMetrics(precisao, recall, f1, acuracia, name_clf,labels):
    df = pd.DataFrame(
        {'classificador': name_clf,
        'precisao': [precisao],
        'recall': [recall],
        'f1': [f1],
        'acuracia': [acuracia],
        'labels':[labels.tolist()]
        }
    )
    df.to_csv(config(name_clf)+'.csv',index=False)



def bagOfWords(X_train, X_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    return X_train,X_test

def trainTeste(X,y,classifier, n, name_clf, labels):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precisao = []
    recall = []
    f1 = []
    accuracia = []
    for i in range(0,n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        X_train,X_test = bagOfWords(X_train,X_test)
        classifier.fit(X_train, y_train) 
        y_pred = classifier.predict(X_test)
        result = precision_recall_fscore_support(y_test, y_pred,labels=labels)
        precisao.append(result[0].tolist())
        recall.append(result[1].tolist())
        f1.append(result[2].tolist())
        accuracia.append(accuracy_score(y_test, y_pred))
    
    saveMetrics(precisao,recall,f1,accuracia,name_clf,labels)
    
def trainTesteMultiple(X,y,classifier, n, name_clf, labels, target):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precisao = []
    recall = []
    f1 = []
    accuracia = []
    for i in range(0,n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        X_train,X_test = bagOfWords(X_train,X_test)
        classifier.fit(X_train, y_train) 
        y_pred = classifier.predict(X_test)
        result = precision_recall_fscore_support(y_test, y_pred,labels=labels)
        precisao.append(result[0].tolist())
        recall.append(result[1].tolist())
        f1.append(result[2].tolist())
        accuracia.append(accuracy_score(y_test, y_pred))
    
    return createDataFrameMetrics(precisao,recall,f1,accuracia,name_clf,labels, target)

def stratifiedShuffleSplitSplits(X,y,classifier, splits,name_clf, labels):
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precisao = []
    recall = []
    f1 = []
    accuracia = []
    sss = StratifiedShuffleSplit(n_splits=splits)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train,X_test = bagOfWords(X_train,X_test)
        classifier.fit(X_train, y_train) 
        y_pred = classifier.predict(X_test)
        result = precision_recall_fscore_support(y_test, y_pred,labels=labels)
        precisao.append(result[0].tolist())
        recall.append(result[1].tolist())
        f1.append(result[2].tolist())
        accuracia.append(accuracy_score(y_test, y_pred))
    
    saveMetrics(precisao,recall,f1,accuracia,name_clf,labels)

def stratifiedShuffleSplitSplitsMultiplus(X,y,classifier, splits,name_clf, labels, target):
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precisao = []
    recall = []
    f1 = []
    accuracia = []
    sss = StratifiedShuffleSplit(n_splits=splits)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train,X_test = bagOfWords(X_train,X_test)
        classifier.fit(X_train, y_train) 
        y_pred = classifier.predict(X_test)
        result = precision_recall_fscore_support(y_test, y_pred,labels=labels)
        precisao.append(result[0].tolist())
        recall.append(result[1].tolist())
        f1.append(result[2].tolist())
        accuracia.append(accuracy_score(y_test, y_pred))
    
    return createDataFrameMetrics(precisao,recall,f1,accuracia,name_clf,labels, target)


def single_round(classifier,X,y):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    X_train,X_test = bagOfWords(X_train,X_test)
        
    classifier.fit(X_train, y_train) 
    y_pred = classifier.predict(X_test)

    print("Matriz de Confusão")
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("Acurácia: ",accuracy_score(y_test, y_pred))

def config(name_clf):
    import os
    from datetime import datetime
    #Coloca o path no diretorio atual
    path = os.path.dirname(os.path.abspath(__file__))

    #Cria a pasta de output caso não exista
    directory_output = 'output'
    path_final = path+'/'+directory_output

    if not os.path.exists(path_final):
        os.makedirs(path_final)

    #Extrai a data e hora atual para colocar no arquivo
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%YT%H_%M_%S")

    #Cria o nome do arquivo onde será gerado o output
    name_csv = name_clf + '_D'+dt_string
    name_csv = path_final+'/'+name_csv
    return name_csv








#name_csv = path_final + '/SVM_D26-11-2019T23_56_19'
#evaluate.evaluate_csv(name_csv)

