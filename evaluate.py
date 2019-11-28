import matplotlib.pyplot as plt
import numpy as np
from statistics import mean,stdev 
import pandas as pd


def get_metrics_multiple_csv(precisao,recall,f1):
    precisao_geral_rodadas = np.mean(precisao, axis=1)
    recall_geral_rodadas = np.mean(recall, axis=1)
    f1_geral_rodadas = np.mean(f1, axis=1)
    return precisao_geral_rodadas,recall_geral_rodadas,f1_geral_rodadas

def showMetrics(precisao, recall, f1, accuracia,labels):
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

def evaluate_csv(name):
    df = pd.read_csv(name+'.csv',encoding ='utf-8')

    p = df.precisao.apply(eval).tolist()
    r= df.recall.apply(eval).tolist()
    f= df.f1.apply(eval).tolist()
    a = df.acuracia.apply(eval).tolist()
    l = df.labels.apply(eval).tolist()

    showMetrics(p[0],r[0],f[0],a[0],l[0])

def show_metrics_multiple_csv(list_precision,list_recall,list_f1,list_acuracia, list_target):
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.75)
    fig.suptitle('Análise Geral')
    axs[0, 0].boxplot(list_acuracia)
    axs[0, 0].set_title('Acurácia')
    axs[0, 0].set_xticklabels(list_target,rotation=35, fontsize=8)
    axs[0, 1].boxplot(list_precision)
    axs[0, 1].set_title('Precisão')
    axs[0, 1].set_xticklabels(list_target,rotation=35, fontsize=8)
    axs[1, 0].boxplot(list_recall)
    axs[1, 0].set_title('Recall')
    axs[1, 0].set_xticklabels(list_target,rotation=35, fontsize=8)
    axs[1, 1].boxplot(list_f1)
    axs[1, 1].set_title('F1')
    axs[1, 1].set_xticklabels(list_target,rotation=35, fontsize=8)
    plt.show()
def evaluate_multiple_csv(csv):
    from ast import literal_eval
    list_precision = []
    list_recall = []
    list_f1 = []
    list_acuracia = []
    list_target = []
    df = pd.read_csv(csv+'.csv',encoding ='utf-8')
    for index, row in df.iterrows():
        p = literal_eval(row.precisao)
        r= literal_eval(row.recall)
        f= literal_eval(row.f1)
        a = literal_eval(row.acuracia)
        lp,lr,lf= get_metrics_multiple_csv(p,r,f)
        list_precision.append(lp)
        list_recall.append(lr)
        list_f1.append(lf)
        list_acuracia.append(a)       
        list_target.append(row.target_csv)
    show_metrics_multiple_csv(list_precision,list_recall,list_f1,list_acuracia,list_target)
