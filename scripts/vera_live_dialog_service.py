from flask import Flask, request, jsonify
import os
import json
import logging
import argparse
import configparser

from deepcubes.models import VeraLiveDialog

config_parser = configparser.RawConfigParser()
config_file_path = os.environ['LIVE_DIALOG_CONF']
config_parser.read(config_file_path)

MODEL_STORAGE = json.loads(
    config_parser.get('live-dialog-service', 'MODEL_STORAGE')
)

EMB_PATH = json.loads(config_parser.get('live-dialog-service', 'EMB_PATH'))

GENETIC_DATA_PATH = json.loads(
    config_parser.get('live-dialog-service', 'GENETIC_DATA_PATH')
)

models = dict()

print("Open log file...")
logging.basicConfig(
    filename='scripts/logs/log.txt',
    format='%(asctime)s %(message)s',
    level=logging.DEBUG
)

print("Prepare app...")
app = Flask(__name__)


def load_model(model_id):
        print("Load model {} ...".format(model_id))
        model_path = os.path.join(
            MODEL_STORAGE, "{}/vera_live_dialog.cube".format(model_id)
        )

        if not os.path.isfile(model_path):
            return False

        if model_id not in models:
            model = VeraLiveDialog.load(model_path)
            models[model_id] = model

        return True


def get_new_model_id(path):
    models_ids = [int(file_name) for file_name in os.listdir(path) if not (
        os.path.isfile(os.path.join(path, file_name))
    )]

    sorted_ids = sorted(models_ids)
    new_model_id = sorted_ids[-1] + 1 if len(sorted_ids) else 0

    return new_model_id


@app.route("/predict", methods=["GET"])
def predict():
    try:
        if (
            request.method not in ["GET", "POST"]
            or ("model_id" not in request.args
                and "model_id" not in request.form)
            or ("query" not in request.args and "query" not in request.form)
        ):
            return jsonify({
                "message": "Please sent GET query with `model_id` and `query` keys",
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
                    "message": "Model with model_id {} not found".format(model_id),
                })

        if 'query' in request.args:
            query = request.args['query']
        elif 'query' in request.form:
            query = request.form['query']
        else:
            return jsonify({"message": "Please send correct json object"})

        if 'labels' in request.args:
            labels = request.args['labels']
        elif 'labels' in request.form:
            query = request.form['labels']
        else:
            labels = list()

        logging.info('predicting intent for question: {}'.format(query))

        model_answer = model(query, labels)

        output = []
        for label, probability in model_answer:
            output.append({
                "label": label,
                "probability": probability,
            })

            logging.info('predicted label: {}'.format(label))
            logging.info('probability: {}'.format(probability))

        return jsonify(output)

    except Exception as e:
        # TODO: FIX. dangerous. loging?
        print(e)


@app.route("/train", methods=["GET", "POST"])
def train():
    try:
        if (
            request.method not in ["GET", "POST"]
            or ("config" not in request.args and "config" not in request.form)
        ):
            return jsonify({
                "message": ("Please sent GET or POST query with `config` key"),
            })

        live_dialog_model = VeraLiveDialog(EMB_PATH, GENETIC_DATA_PATH)

        # parse data from json
        if 'config' in request.args:
            config = json.loads(request.args["config"])
        elif 'config' in request.form:
            config = json.loads(request.form["config"])
        else:
            return jsonify({"message": "Please send correct json object"})

        live_dialog_model.train(config)

        new_model_id = get_new_model_id(MODEL_STORAGE)
        models[new_model_id] = live_dialog_model

        model_path = os.path.join(MODEL_STORAGE, str(new_model_id))
        live_dialog_model.save(model_path)

        logging.info('saved model with model_id {}'.format(new_model_id))

        return jsonify({
            "message": 'Created model with model_id {}'.format(new_model_id),
            "model_id": new_model_id,
        })

    except Exception as e:
        # TODO: FIX. dangerous. loging?
        print(e)


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
