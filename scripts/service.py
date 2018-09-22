from flask import Flask, request, jsonify
import os
import json
import logging

from intentclf.models import Embedder
from intentclf.models import IntentClassifier

models_storage = 'scripts/models/'

print("Open log file...")
logging.basicConfig(
    filename='scripts/logs/log.txt',
    format='%(asctime)s %(message)s',
    level=logging.DEBUG
)

print("Load embedder...")
embedder = Embedder(os.environ['INTENT_CLASSIFIER_MODEL'])

print("Prepare app...")
app = Flask(__name__)


def get_sorted_models_ids(path):
    models = [
        file_name for file_name in os.listdir(models_storage) if (
            'pickle' in file_name
        )
    ]

    models_ids = map(
        lambda m: int(m.split('model-')[1].split('.pickle')[0]),
        models
        )
    return sorted(models_ids)


@app.route("/answer", methods=["GET"])
def answer():
    if (
        request.method != "GET" or
        "id" not in request.args or
        "question" not in request.args
    ):
        return jsonify({
            "message": "Please sent GET query with `id` and `question` keys",
        })

    classifier = IntentClassifier(embedder)

    model_id = request.args.get("id")

    if int(model_id) in get_sorted_models_ids(models_storage):
        classifier.load(
            "scripts/models/model-{}.pickle".format(model_id)
        )
    else:
        return jsonify({
            "message": "Model with id {} not found".format(model_id),
        })

    logging.info(
        'predicting intent for question: {}'.format(
            request.args.get("question")
        )
    )
    answer = classifier.predict(request.args.get("question"))

    logging.info('predicted intent: {}'.format(answer))

    return jsonify({
        "answer": answer,
    })


@app.route("/train", methods=["GET", "POST"])
def train():
    if (
        request.method not in ["GET", "POST"] or
        "intent_json" not in request.args
    ):
        return jsonify({
            "message": "Please sent GET or POST query with `intent_json` keys",
        })

    intentclf = IntentClassifier(embedder)
    questions, answers = [], []

    # parse data from json
    try:
        data = json.loads(request.args["intent_json"])
    except:
        return jsonify({
            "message": "Please send correct json object"
        })

    for label, category in enumerate(data):
        answer = category["answers"][0]

        for question in category["questions"]:
            questions.append(question)
            answers.append(answer)

    if len(data) < 2:
        return jsonify({
            "message": "For training the model requires 2 or more intents",
        })

    intentclf.train(questions, answers)

    new_model_id = get_sorted_models_ids(models_storage)[-1] + 1
    intentclf.save(models_storage + 'model-{}.pickle'.format(new_model_id))
    logging.info('saved model with id {}'.format(new_model_id))

    return jsonify({
        "message": 'Created model with id {}'.format(new_model_id),
        "id": new_model_id,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3335, debug=False)
