from flask import Flask, request, jsonify
import os
import sys
import json
import logging
import argparse
import configparser

from deepcubes.models import LogisticIntentClassifier
from deepcubes.models import MultistagIntentClassifier

logger = logging.getLogger("MultistageClassifierService")
logger.setLevel(logging.INFO)

# create the logging file handler
fh = logging.FileHandler("scripts/logs/multistage_classifier_service.log")
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("Started Multistage Classifier Server...")

if 'SERVICE_CONF' in os.environ:
    config_file_path = os.environ['SERVICE_CONF']
else:
    logger.warning('Config file not found. Test config is used...')
    config_file_path = 'tests/data/test.conf'

logger.info("Read config file {} ...".format(config_file_path))
config_parser = configparser.RawConfigParser()
config_parser.read(config_file_path)


MODEL_STORAGE = json.loads(
    config_parser.get('multistage-classifier-service', 'MODEL_STORAGE')
)
logger.info("Model storage: {} ...".format(MODEL_STORAGE))

MAJOR_MODEL_ID = json.loads(
    config_parser.get('multistage-classifier-service', 'MAJOR_MODEL_ID')
)

MINOR_MODEL_ID = json.loads(
    config_parser.get('multistage-classifier-service', 'MINOR_MODEL_ID')
)

GROUPS_DATA_PATH = json.loads(
    config_parser.get('multistage-classifier-service', 'GROUPS_DATA_PATH')
)
logger.info("Groups csv file: {} ...".format(GROUPS_DATA_PATH))

logger.info("Prepare Flask app...")
app = Flask(__name__)


def load_model(major_model_id, minor_model_id, groups_data_path):
    logger.info("Loading major model {} ...".format(major_model_id))
    major_model_path = os.path.join(
        MODEL_STORAGE, "{}/intent_classifier.cube".format(major_model_id)
    )
    if not os.path.isfile(major_model_path):
        logger.error(
            "Model {} not found".format(major_model_id)
        )
        return False

    logger.info("Loading minor model {} ...".format(minor_model_id))
    minor_model_path = os.path.join(
        MODEL_STORAGE, "{}/intent_classifier.cube".format(minor_model_id)
    )
    if not os.path.isfile(minor_model_path):
        logger.error(
            "Model {} not found".format(minor_model_id)
        )
        return False

    major_model = LogisticIntentClassifier.load(major_model_path)
    minor_model = LogisticIntentClassifier.load(minor_model_path)
    multistage_model = MultistagIntentClassifier(major_model, minor_model)
    multistage_model.train(groups_data_path)

    return multistage_model


@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if (
            request.method not in ["GET", "POST"]
            or ("query" not in request.args and "query" not in request.form)
        ):
            logger.error("Received invalid request")
            return jsonify({
                "message": ("Please sent GET or POST query"
                            "with `query` key"),
            })

        logger.info("Received {} request from {}".format(request.method,
                                                         request.remote_addr))

        # parse data from json
        if 'query' in request.args:
            query = request.args['query']
        elif 'query' in request.form:
            query = request.form['query']
        else:
            logger.error("Received invalid json object")
            return jsonify({"message": "Please send correct json object"})

        logger.info('Received query: {}'.format(query))

        answer = multistage_model(query)
        output = [{
            "answer": answer,
            "probability": None,
            "threshold": None,
            "accuracy_score": None
        }]
        logger.info("Top predicted label: {}".format(answer))

        logger.info("Sending response...")
        return jsonify(output)

    except Exception:
        _, exc_obj, exc_tb = sys.exc_info()
        logger.error("line {}, {}".format(exc_tb.tb_lineno, exc_obj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Launch Multistage Intent Classifier API'
    )

    parser.add_argument(
        '-major', '--major_model_id',
        help='Major model id for intent classifier', default=MAJOR_MODEL_ID
    )
    parser.add_argument(
        '-minor', '--minor_model_id',
        help='Minor model id for intent classifier', default=MINOR_MODEL_ID
    )
    parser.add_argument(
        '-groups', '--groups_data_path',
        help='Data path to csv with intent groups', default=GROUPS_DATA_PATH
    )

    args = parser.parse_args()

    multistage_model = load_model(
        args.major_model_id, args.minor_model_id, args.groups_data_path
    )
    if multistage_model:
        app.run(host="0.0.0.0", port=3339, debug=False)
    else:
        logger.error("Failed to download models")
