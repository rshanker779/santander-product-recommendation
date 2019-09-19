from flask import Flask
from flask import request
import santander_kaggle.internal_api as api
from rshanker779_common.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)


@app.route("/")
@app.route("/index", methods=["GET"])
def index():
    return "Santander recommendation model. Send POST requests to /predict"


@app.route("/predict", methods=["POST"])
def get_prediction():
    logger.info("Handling request %s with form %s", request, request.form)
    logger.info(request.form)
    return api.handler.handle_json_request(request.get_json())


if __name__ == "__main__":
    app.run(host="0.0.0.0")
