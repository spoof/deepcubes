from flask import Flask, request, jsonify
import os
import json
import logging
import argparse
import configparser

from deepcubes.models import LogisticIntentClassifier

if 'SERVICE_CONF' in os.environ:
    config_file_path = os.environ['SERVICE_CONF']
else:
    print('Config file not found. Text config is used...')
    config_file_path = 'tests/data/test.conf'

config_parser = configparser.RawConfigParser()
config_parser.read(config_file_path)

MODEL_STORAGE = json.loads(
    config_parser.get('classifier-service', 'MODEL_STORAGE')
)

models = dict()

print("Open log file...")
logging.basicConfig(
    filename='scripts/logs/classifier_service_log.txt',
    format='%(asctime)s %(message)s',
    level=logging.DEBUG
)

print("Prepare app...")
app = Flask(__name__)


def load_model(model_id):
    print("Load model {} ...".format(model_id))
    model_path = os.path.join(
        MODEL_STORAGE, "{}/intent_classifier.cube".format(model_id)
    )
    if not os.path.isfile(model_path):
        return False

    if model_id not in models:
        model = LogisticIntentClassifier.load(model_path)
        models[model_id] = model

    return True


def get_new_model_id(path):
    models_ids = [int(file_name) for file_name in os.listdir(path) if not (
        os.path.isfile(os.path.join(path, file_name))
    )]

    sorted_ids = sorted(models_ids)
    new_model_id = sorted_ids[-1] + 1 if len(sorted_ids) else 0

    return new_model_id


@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if (
            request.method not in ["GET", "POST"]
            or ("model_id" not in request.args
                and "model_id" not in request.form)
            or ("query" not in request.args and "query" not in request.form)
        ):
            return jsonify({
                "message": ("Please sent GET query"
                            "with `model_id` and `query` keys"),
            })

        # parse data from json
        if 'model_id' in request.args:
            model_id = int(request.args['model_id'])
        elif 'model_id' in request.form:
            model_id = int(request.form['model_id'])
        else:
            return jsonify({"message": "Please send correct json object"})

        if model_id in models:
            model = models[model_id]
        else:
            loading = load_model(model_id)
            if loading:
                model = models[model_id]
            else:
                return jsonify({
                    "message": "Model with model_id {} not found".format(
                        model_id),
                })

        if 'query' in request.args:
            query = request.args['query']
        elif 'query' in request.form:
            query = request.form['query']
        else:
            return jsonify({"message": "Please send correct json object"})

        logging.info('predicting intent for question: {}'.format(query))

        model_answer = model(query)

        output = []
        for label, probability in model_answer:
            output.append({
                "label": label,
                "proba": probability,
                "threshold": 0.3,
                "accuracy_score": None
            })

            logging.info('predicted label: {}'.format(label))
            logging.info('probability: {}'.format(probability))

        return jsonify(output)

    except Exception as e:
        # TODO: FIX. dangerous. loging?
        print(repr(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Launch Vera Live Dialog API'
    )

    parser.add_argument(
        '-m', '--model_id_list', nargs='+', type=int, default=list()
    )

    args = parser.parse_args()
    for model_id in args.model_id_list:
        loading = load_model(model_id)

        if not loading:
            print("Model with id {} not found".format(model_id))

    app.run(host="0.0.0.0", port=3339, debug=False)
