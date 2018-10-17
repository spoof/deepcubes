from flask import Flask, request, jsonify
import os
import json
import logging

from intentclf.models import Embedder
from intentclf.models import IntentClassifier

MODEL_STORAGE = 'scripts/models/'
TRASH_QUESTIONS_PATH = 'scripts/data/trash_questions.csv'

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

    try:
        classifier.load(model_id, MODEL_STORAGE)
    except FileNotFoundError:
        return jsonify({
            "message": "Model with id {} not found".format(model_id),
        })

    logging.info(
        'predicting intent for question: {}'.format(
            request.args.get("question")
        )
    )
    answer, probability = classifier.predict(
        request.args.get("question").strip()
    )
    if classifier.label_to_accuracy_score:
        label = classifier.answer_to_label[answer]
        accuracy_score = classifier.label_to_accuracy_score[label]
    else:
        accuracy_score = None
    logging.info('predicted intent: {}'.format(answer))

    return jsonify({
        "answer": answer,
        "probability": probability,
        "threshold": classifier.threshold,
        "accuracy_score": accuracy_score,
    })


@app.route("/train", methods=["GET", "POST"])
def train():
    if (
        request.method not in ["GET", "POST"] or
        ("intent_json" not in request.args and
            "intent_json" not in request.form)
    ):
        return jsonify({
            "message": "Please sent GET or POST query with `intent_json` keys",
        })

    intentclf = IntentClassifier(embedder)
    questions, answers = [], []

    # parse data from json
    if 'intent_json' in request.args:
        data = json.loads(request.args["intent_json"])
    elif 'intent_json' in request.form:
        data = json.loads(request.form["intent_json"])
    else:
        return jsonify({
            "message": "Please send correct json object"
        })
    for label, category in enumerate(data):
        answer = category["answers"][0]

        for question in category["questions"]:
            questions.append(question.strip())
            answers.append(answer.strip())

    if len(data) < 2:
        return jsonify({
            "message": "For training the model requires 2 or more intents",
        })

    intentclf.train(questions, answers)
    if os.path.isfile(TRASH_QUESTIONS_PATH):
        intentclf.threshold_calc(TRASH_QUESTIONS_PATH)
    else:
        intentclf.threshold_calc()
    new_model_id = intentclf.save(MODEL_STORAGE)
    logging.info('saved model with id {}'.format(new_model_id))

    return jsonify({
        "message": 'Created model with id {}'.format(new_model_id),
        "id": new_model_id,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3335, debug=False)
