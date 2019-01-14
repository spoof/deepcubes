from flask import Flask, request, jsonify
import os
import json
import logging
import argparse

from deepcubes.models import VeraLiveDialog

MODEL_STORAGE = 'scripts/models/'

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
            MODEL_STORAGE, "{}/live_dialog.cube".format(model_id)
        )
        if not os.path.isfile(model_path):
            return False
        if model_id not in models:
            model = VeraLiveDialog.load(model_path)
            models[model_id] = model
        return True


@app.route("/predict", methods=["GET"])
def predict():
    if (
        request.method != "GET" or
        "id" not in request.args or
        "query" not in request.args
    ):
        return jsonify({
            "message": "Please sent GET query with `id` and `query` keys",
        })

    model_id = int(request.args.get("id"))

    if model_id in models:
        model = models[model_id]
    else:
        loading = load_model(model_id)
        if loading:
            model = models[model_id]
        else:
            return jsonify({
                "message": "Model with id {} not found".format(model_id),
            })

    query = request.args.get("query")
    if 'labels' in request.args:
        labels = request.args.get("labels")
    else:
        labels = list()
    logging.info('predicting intent for question: {}'.format(query))

    model_answer = model.predict(query, labels)

    output = []
    for label, probability in model_answer:
        output.append({
            "label": label,
            "probability": probability,
        })

        logging.info('predicted label: {}'.format(label))
        logging.info('probability: {}'.format(probability))

    return jsonify(output)


@app.route("/train", methods=["GET", "POST"])
def train():
    if (
        request.method not in ["GET", "POST"] or
        ("config" not in request.args and
            "config" not in request.form)
    ):
        return jsonify({
            "message": ("Please sent GET or POST query with `config` key"),
        })

    live_dialog_model = VeraLiveDialog()

    # parse data from json
    if 'config' in request.args:
        config = json.loads(request.args["config"])
    elif 'config' in request.form:
        config = json.loads(request.form["config"])
    else:
        return jsonify({
            "message": "Please send correct json object"
        })

    live_dialog_model.train(config)

    new_model_id = live_dialog_model.save(MODEL_STORAGE)
    logging.info('saved model with id {}'.format(new_model_id))

    return jsonify({
        "message": 'Created model with id {}'.format(new_model_id),
        "id": new_model_id,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Launch vera live dialog API'
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
