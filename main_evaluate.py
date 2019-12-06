import evaluate

def evaluate_multiple_csv(name_file):
    import os
    #Coloca o path no diretorio atual
    path = os.path.dirname(os.path.abspath(__file__))
    directory_output = 'output'
    path_final = path+'/'+directory_output+'/'
    evaluate.evaluate_multiple_csv(path_final+name_file)

def evaluate_multiple_classifiers(name_file):
    import os
    #Coloca o path no diretorio atual
    path = os.path.dirname(os.path.abspath(__file__))
    directory_output = 'output'
    path_final = path+'/'+directory_output+'/'
    #evaluate.evaluate_multiple_classifiers_by_class(path_final+name_file)
    evaluate.evaluate_multiple_classifiers_means(path_final+name_file)

def evaluate_csv(name_file):
    import os
    #Coloca o path no diretorio atual
    path = os.path.dirname(os.path.abspath(__file__))
    directory_output = 'output'
    path_final = path+'/'+directory_output+'/'
    evaluate.evaluate_csv(path_final+name_file)

if __name__ == "__main__":
    file_name = 'NFR_MULTIPLE_CLASSIFIERS_D06-12-2019T17_56_12'
    #evaluate_multiple_csv(file_name)
    evaluate_multiple_classifiers(file_name)
    #evaluate_csv(file_name)