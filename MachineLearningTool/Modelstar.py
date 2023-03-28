import sys
import os
from datetime import datetime


# sys.path.append('/home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction')
def Setparameter(serverset, datapath, labelpath, covariatespath, parametersetting):
    '''
    parameter setting
    :return:dict


    '''
    mark = parametersetting['mark']
    pathname = datapath.split('/')
    projectname =pathname[pathname.index('Project')+1]
    pathbox = datapath[0:datapath.find('Project') + 8]
    sersavepath = pathbox + projectname + '/Res/' +mark+'_Res/server_log/'
    weightpath = pathbox + projectname + '/Res/' +mark+'_Res/model_weight/'
    outputdatapath = pathbox + projectname + '/Res/' + mark + '_Res/output_data/'
    scriptpath = pathbox + projectname + '/Res/' +mark+'_Res/script/'+mark+'_script/'
    PATH = [sersavepath, weightpath, scriptpath, outputdatapath]
    for p in PATH:
        if not os.path.exists(p):
            os.makedirs(p)
    directory_path = os.path.dirname(os.path.abspath(__file__))
    print(directory_path)
    Modelcodepath = directory_path+'/'+parametersetting['Modelcodefile']


    dimention = parametersetting['dimention']            # General Ext ADHD Int Age Reflection TAI BIS
    permutation = parametersetting['permutation']        # 1: Permutation test   0: no
    kfold = parametersetting['kfold']                    # number:KFold 0:no
    CVRepeatTimes = parametersetting['CVRepeatTimes']
    # dataMark = parametersetting['Mark']
    CovariatesMark = parametersetting['CovariatesMark']  # 1 :do   0: no
    Time = parametersetting['Time']                      # 0 : test

    setparameter = {
            'serverset':         serverset,
          'sersavepath':       sersavepath,
           'scriptpath':        scriptpath,
        'CVRepeatTimes':     CVRepeatTimes,
             'datapath':          datapath,
            'labelpath':         labelpath,
           'weightpath':        weightpath,
            'dimention':         dimention,
          'Permutation':       permutation,
       'covariatespath':    covariatespath,
                 'Time':              Time,
                'KFold':             kfold,
       'CovariatesMark':    CovariatesMark,
             'dataMark':              mark,
        'Modelcodepath':     Modelcodepath,
       'outputdatapath':     outputdatapath
    }
    return setparameter

def PLSc_RandomCV_MultiTimes(serverset, sersavepath, scriptpath, CVRepeatTimes, kfold, dimention, Modelcodepath, Permutation=0):
    '''
    :param serverset: Server parameter settings
    :param savepath:The result storage path of the server
    :param scriptpath:The path to the script on the server
    :param CVRepeatTimes:Script execution times
    :param Permutation: Whether to replace the test  1：permutation test 0：no permutation test
    :return:
    '''
#    Sbatch_Para = '#!/bin/bash\n'+'#SBATCH --qos=high_c\n'+'#SBATCH --job-name={}\n#SBATCH --nodes={}\n#SBATCH --ntasks={}\n#SBATCH --cpus-per-task={}\n#SBATCH --mem-per-cpu={}\n#SBATCH -p {}\n'.format(*serverset)
    Sbatch_Para = '#!/bin/bash\n'+'#SBATCH --job-name={}\n#SBATCH --nodes={}\n#SBATCH --ntasks={}\n#SBATCH --cpus-per-task={}\n#SBATCH --mem-per-cpu={}\n#SBATCH -p {}\n'.format(*serverset)

    if kfold:
       system_cmd = 'python' +' '+Modelcodepath
    else:
       system_cmd = 'python' +' '+Modelcodepath
    if Permutation == 0:
        scriptfold = scriptpath + '/' + str(datetime.now().strftime('%Y_%m_%d'))+'_'+ dimention
        if os.path.exists(scriptfold):
            print('WARNING!!!!\n')
            print('you have not delete script [Time_*_.script.sh]! in \n', scriptfold)
            return
        if not os.path.exists(scriptfold):
            os.makedirs(scriptfold)

        servernotepath = sersavepath + str(datetime.now().strftime('%Y_%m_%d'))+'_'+dimention
        if not os.path.exists(servernotepath):
            os.makedirs(servernotepath)

        for i in range(1, CVRepeatTimes+1):
            script = open(scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh', mode='w')
            script.write(Sbatch_Para)
            script.write('\n')
            script.write('#SBATCH -o ' + servernotepath + '/'+'Time_' + str(i) + '_' + 'job.%j.out\n')
            script.write('#SBATCH -e ' + servernotepath + '/'+'Time_' + str(i) + '_' + 'job.%j.error.txt\n\n')
            script.write(system_cmd +' '+str(i))
            script.close()
            os.system('chmod +x ' + scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh')

            os.system('sbatch ' + scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh')
    else:
        scriptfold = scriptpath + '/' + str(datetime.now().strftime('%Y_%m_%d')) + '_' + dimention
        if os.path.exists(scriptfold):
            print('WARNING!!!!\n')
            print('you have not delete script [Time_*_.script.sh]! in \n', scriptfold)
            return
        if not os.path.exists(scriptfold):
            os.makedirs(scriptfold)

        servernotepath = sersavepath + str(datetime.now().strftime('%Y_%m_%d')) + '_' + dimention
        if not os.path.exists(servernotepath):
            os.makedirs(servernotepath)
        for i in range(1, CVRepeatTimes + 1):
            script = open(scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh', mode='w')
            script.write(Sbatch_Para)
            script.write('\n')
            script.write('#SBATCH -o ' + servernotepath + '/' + 'Time_' + str(i) + '_' + 'job.%j.out\n')
            script.write('#SBATCH -e ' + servernotepath + '/' + 'Time_' + str(i) + '_' + 'job.%j.error.txt\n\n')
            script.write(system_cmd + ' ' + str(i))
            script.close()
            os.system('chmod +x ' + scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh')

            os.system('sbatch ' + scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh')

datapath = '/home/cuizaixu_lab/fanqingchen/DATA_C/Project/HCPD/data/HCPDMori68FC.txt'  # feture matrix
labelpath = '/home/cuizaixu_lab/fanqingchen/DATA_C/Project/HCPD/label/flanker01_label.csv'
covariatespath = '/home/cuizaixu_lab/fanqingchen/DATA_C/Project/HCPD/label/covariates.csv'





# the path of saving file **
datapath = 'your data path'  # feture matrix
labelpath = 'your label path'
covariatespath = 'your covariates path'

serverset = ['fanqingchen', 1, 1, 3, 10000, 'q_fat_c']                    # name=fanqingchen | nodes=1 | ntasks=1| cpus-per-task=3 |mem-per-cpu=10000| -p q_fat_c
parametersetting = {
       'permutation': 0,           # 1: Permutation test   0: no
             'kfold': 2,           # number:KFold 0:no
     'CVRepeatTimes': 1000,
              'mark': 'dataMark',
    'CovariatesMark': 0,           # 1 :do   0: no
              'Time': 20230310,
            'method': 'PLSR',      #    
     'Modelcodefile': 'Machinelearning_model.py'
    }

spar = Setparameter(serverset, datapath, labelpath, covariatespath, parametersetting)

PLSc_RandomCV_MultiTimes(
                    spar['serverset'],
                    spar['sersavepath'],
                    spar['scriptpath'],
                    spar['CVRepeatTimes'],
                    spar['KFold'],
                    spar['dimention'],
                    spar['Modelcodepath'],
                    spar['Permutation']
                        )


