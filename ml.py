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


def boxplot_tipo1(precisao_classe,recall_classe,f1_classe,precisao_geral_rodadas,recall_geral_rodadas,f1_geral_rodadas,accuracia,labels):
    
    precisao_grupos = list(precisao_classe)
    precisao_grupos.append(precisao_geral_rodadas)
    recall_grupos = list(recall_classe)
    recall_grupos.append(recall_geral_rodadas)
    f1_grupos = list(f1_classe)
    f1_grupos.append(f1_geral_rodadas)
    
    
    labels_grupos = list(labels)
    labels_grupos.append("GERAL")

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Análise Geral')
    axs[0, 0].boxplot(accuracia)
    axs[0, 0].set_title('Acurácia')
    axs[0, 1].boxplot(precisao_grupos)
    axs[0, 1].set_xticklabels(labels_grupos)
    axs[0, 1].set_title('Precisão')
    axs[1, 0].boxplot(recall_grupos)
    axs[1, 0].set_title('Recall')
    axs[1, 1].boxplot(f1_grupos)
    axs[1, 1].set_title('F1')
    plt.show()


def showMetrics(precisao, recall, f1, accuracia,labels):
    import numpy as np
    from statistics import mean,stdev 
    
    precisao_classe = []
    for i in range(len(precisao[0])):
        precisao_classe.append([class_value[i] for class_value in precisao])
    print("Array Precisao: ", precisao)
    print("Precisao por classe: ", precisao_classe)

    recall_classe = []
    for i in range(len(recall[0])):
        recall_classe.append([class_value[i] for class_value in recall])
    print("Array Recall: ", recall)
    print("Recall por classe: ", recall_classe)

    f1_classe = []
    for i in range(len(f1[0])):
        f1_classe.append([class_value[i] for class_value in f1])
    print("Array F1: ", f1)
    print("F1 por classe: ", f1_classe)

    precisao_geral_classe = np.mean(precisao, axis=0)
    print("Precisão Geral por classe: ", precisao_geral_classe)
    precisao_geral = mean(precisao_geral_classe)
    print("Precisão: %0.3f (+/- %0.3f)" % (mean(precisao_geral_classe), stdev(precisao_geral_classe) * 2))
    precisao_geral_rodadas = np.mean(precisao, axis=1)
    print("Precisão geral rodadas: ",precisao_geral_rodadas)
    
    
    recall_geral_classe = np.mean(recall, axis=0)
    print("Recall Geral por classe: ",recall_geral_classe)
    recall_geral = mean(recall_geral_classe)
    print("Recall: %0.3f (+/- %0.3f)" % (mean(recall_geral_classe), stdev(recall_geral_classe) * 2))
    recall_geral_rodadas = np.mean(recall, axis=1)
    print("Recall geral rodadas: ",recall_geral_rodadas)

    f1_geral_classe = np.mean(f1, axis=0)
    print("F1 Geral por classe: ",f1_geral_classe)
    f1_geral = mean(f1_geral_classe)
    print("F1: %0.3f (+/- %0.3f)" % (mean(f1_geral_classe), stdev(f1_geral_classe) * 2))
    f1_geral_rodadas = np.mean(f1, axis=1)
    print("F1 geral rodadas: ",f1_geral_rodadas)

    accuracia_geral = mean(accuracia)
    print("Acurácia: %0.3f (+/- %0.3f)" % (mean(accuracia), stdev(accuracia) * 2))
    print("Acurácia rodadas:", accuracia)

    boxplot_tipo1(precisao_classe,recall_classe,f1_classe,precisao_geral_rodadas,recall_geral_rodadas,f1_geral_rodadas,accuracia,labels)

    
def saveMetrics(precisao, recall, f1, acuracia, name_clf,labels,name_csv):
    df = pd.DataFrame(
        {'classificador': name_clf,
        'precisao': [precisao],
        'recall': [recall],
        'f1': [f1],
        'acuracia': [acuracia],
        'labels':[labels.tolist()]
        }
    )
    df.to_csv(name_csv+'.csv',index=False)



def bagOfWords(X_train, X_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    return X_train,X_test

def trainTeste(X,y,classifier, n, name_clf, name_csv):
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
    #showMetrics(precisao,recall,f1,accuracia)
    saveMetrics(precisao,recall,f1,accuracia,name_clf,labels, name_csv)
    
   

def stratifiedShuffleSplitSplits(X,y,classifier, splits):
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
        result = precision_recall_fscore_support(y_test, y_pred)
        precisao.append(result[0])
        recall.append(result[1])
        f1.append(result[2])
        accuracia.append(accuracy_score(y_test, y_pred))
    
    showMetrics(precisao,recall,f1,accuracia)


def evaluate_csv(name):
    df = pd.read_csv(name+'.csv',encoding ='utf-8')

    p = df.precisao.apply(eval).tolist()
    r= df.recall.apply(eval).tolist()
    f= df.f1.apply(eval).tolist()
    a = df.acuracia.apply(eval).tolist()
    l = df.labels.apply(eval).tolist()

    showMetrics(p[0],r[0],f[0],a[0],l[0])

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


import os
from datetime import datetime
#Coloca o path no diretorio atual
path = os.path.dirname(os.path.abspath(__file__))

#Cria a pasta de output caso não exista
directory_output = 'output'
path_final = path+'/'+directory_output

if not os.path.exists(path_final):
    os.makedirs(path_final)

#define o nome do arquivo csv a ser lido. Define o nome da técnica que sera utilizada
filee = 'nfr_dataset'
name_clf = 'SVM'

#Extrai a data e hora atual para colocar no arquivo
now = datetime.now()
dt_string = now.strftime("%d-%m-%YT%H_%M_%S")

#Cria o nome do arquivo onde será gerado o output
name_csv = name_clf + '_D'+dt_string
name_csv = path_final+'/'+name_csv


#Leitura da Planilha
path_to_file = path+'/datasets/'+filee
df = pd.read_csv(path_to_file+'.csv',encoding ='utf-8')

#overview(df)

#Change Columns - F and NF
df['class'] = np.where(df['class'] == 'F', 'F','RNF')

#SELECT ONLY WITH VALUES
#df = df.loc[df['class'].isin(['US','SE','O','PE'])]
#df= df.reset_index()

#Separacao das features e dos targets (X,Y)
X, y = df.requirement_text, df.iloc[:,-1]
#print(y)

#Extrai as LABELS do dataset
from sklearn.utils.multiclass import unique_labels
labels = unique_labels(y)
#print("LABELS: ",labels)

#Definição do Classificador
from sklearn import svm
classifier = svm.SVC(gamma='scale')

#from sklearn.naive_bayes import MultinomialNB
#classifier = MultinomialNB()


#Chamadas
#single_round(classifier,X,y)
#stratifiedShuffleSplitSplits(X,y,classifier,10)
#trainTeste(X,y,classifier,2,name_clf,name_csv)

#name_csv = path_final + '/SVM_D26-11-2019T23_56_19'
#evaluate_csv(name_csv)

