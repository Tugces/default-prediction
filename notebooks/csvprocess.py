from flask import Response
import io

def downloadCSV(predictedDataFinal):
    """
    This function is created to download prediction results in the requested format.
    """
    file_buffer = io.StringIO()
    predictedDataFinal[['uuid', 'pd']].to_csv(file_buffer, encoding="utf-8", index=False, sep=",")
    file_buffer.seek(0)
    response = Response(file_buffer, mimetype="text/csv")
    # add a filename
    response.headers.set(
        "Content-Disposition", "attachment", filename="{0}.csv".format("defaultPredictions")
    )
    return response