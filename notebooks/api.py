from flask import Flask
import variables
import dataprocess
import model
import csvprocess


app = Flask(__name__)

@app.route('/predictions/', methods=['GET'])
def getPredictions():
    """
    This function is created to run all the processes and the pickles which were observed during the model training.
    The variables are defined in the variables.
    """
    data = dataprocess.getData(variables.path)
    dataPredict = dataprocess.getPredictData(variables.target, data)
    dataTrain = dataprocess.getTrainData(variables.target, data)
    dataProcessed = dataprocess.fillingMissingValues(dataTrain, dataPredict, variables.target)
    dataProcessConcat = dataprocess.runOhe(dataTrain, dataProcessed)
    predictedDataFinal = model.runModel(dataProcessConcat, variables.feature_list)
    return csvprocess.downloadCSV(predictedDataFinal)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')