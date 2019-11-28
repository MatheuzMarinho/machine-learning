import evaluate

def evaluate_multiple_csv(name_file):
    import os
    #Coloca o path no diretorio atual
    path = os.path.dirname(os.path.abspath(__file__))
    directory_output = 'output'
    path_final = path+'/'+directory_output+'/'
    evaluate.evaluate_multiple_csv(path_final+name_file)

def evaluate_csv(name_file):
    import os
    #Coloca o path no diretorio atual
    path = os.path.dirname(os.path.abspath(__file__))
    directory_output = 'output'
    path_final = path+'/'+directory_output+'/'
    evaluate.evaluate_csv(path_final+name_file)

if __name__ == "__main__":
    file_name = 'SVM_DIARIO_MUTIPLE_LABELS_D27-11-2019T23_30_06'
    evaluate_multiple_csv(file_name)
    #evaluate_csv(file_name)