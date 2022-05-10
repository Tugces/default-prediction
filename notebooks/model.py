import pickle

def runModel(data, feature_list):
    """
    This function is created to fget the pickle of the trained model.
    """
    #model pickle
    pickled_model = pickle.load(open('../app/models/final_model.pkl', 'rb'))
    predictionArray = pickled_model.predict_proba(data[feature_list])
    data['pd'] = [item[1] for item in predictionArray]
    return data