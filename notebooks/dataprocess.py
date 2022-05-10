import modules.missingimputation as missingimputation
from modules.ohe_functions import oheConcat, oheTransform
import pickle
import pandas as pd

def getData(path):
    """
    This function is created to read the csv data.
    """
    data = pd.read_csv(path, delimiter=';')
    return data

def getPredictData(target, data):
    """
    This function is created to filter the data without target.
    """
    dataPredict = data[data[target].isna()].reset_index(drop = True)
    return dataPredict

def getTrainData(target, data):
    """
    This function is created to get the train data.
    """
    dataTrain = data[~data[target].isna()].reset_index(drop = True)
    return dataTrain

def fillingMissingValues(dataTrain, dataPredict, target):
    """
    This function is created to fill the missing values in the dataPredict according to the analysis in dataTrain data.
    """
    for col in dataPredict.columns:
        # define lists
        colList_fillZero = ['account_days_in_dc_12_24m', 'account_days_in_rem_12_24m', 'account_days_in_term_12_24m']
        colList_fillMean = ['avg_payment_span_0_12m', 'num_active_div_by_paid_inv_0_12m']
        colList_fillMode = ['num_arch_written_off_0_12m', 'num_arch_written_off_12_24m']

        # Run missingImputationZero for the predict set
        if col in colList_fillZero:
            missingimputation.missingImputationZero(dataPredict, col)
        # Run missingImputationMean for the predict set
        elif col in colList_fillMean:
            missingimputation.missingImputationMean(dataTrain, dataPredict, target, col)
        # Run missingImputationMode for the predict set
        elif col in colList_fillMode:
            missingimputation.missingImputationMode(dataTrain, dataPredict, target, col)
            
    return dataPredict


def runOhe(dataTrain, dataProcessed):
    """
    This function is created to get the pickle of OneHotEncoder, it is performed same as training period.
    """
    pickled_ohe = pickle.load(open('../app/models/ohe.pkl', 'rb'))
    catColumns = dataTrain.select_dtypes(include=[object, bool]).iloc[:,1:].columns
    dataPredictOhe = oheTransform(ohe = pickled_ohe, data = dataProcessed, catColumns = catColumns)
    dataPredictConcat = oheConcat(data=dataProcessed, data_ohe = dataPredictOhe, catColumns = catColumns)
    return dataPredictConcat