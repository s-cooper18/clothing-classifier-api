import logging

import azure.functions as func
from PIL import Image
import io
import json


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    bin_image = req.get_body()
    if not bin_image:
        try:
            req_body = req.get_body()
        except ValueError:
            pass
        else:
            bin_image = req_body.get('image')

    try:
        image = Image.open(io.BytesIO(bin_image))
        print("returned successfully")
    except IOError:
        return func.HttpResponse(
            "Bad input. Unable to cast request body to an image format.",
            status_code=400
        )

    # classification
    categories = {"classification": "dog"}
    # categories = classify_function

    # Convert to json object to send
    try:
        json_object = json.dumps(categories, indent = 4)  
        print("returned successfully")
    except IOError:
        return func.HttpResponse(
            "Unable to convert to json",
            status_code=400
        )

    # Return category
    return func.HttpResponse(
        json_object,
        status_code=200)
