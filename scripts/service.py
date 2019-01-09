from flask import Flask, request, jsonify
import os
import json
import pickle
import logging
import argparse

from deepcubes.cubes import Embedder
from deepcubes.cubes import IntentClassifier

MODEL_STORAGE = 'scripts/models/'
TRASH_QUESTIONS_PATH = 'scripts/data/trash_questions.csv'

embedders = dict()

print("Open log file...")
logging.basicConfig(
    filename='scripts/logs/log.txt',
    format='%(asctime)s %(message)s',
    level=logging.DEBUG
)

print("Load default embedder...")
default_embedders = {
    'ru': Embedder(os.environ['INTENT_CLASSIFIER_MODEL']),
    'en': Embedder(os.environ['INTENT_CLASSIFIER_ENG_MODEL'])
}

print("Prepare app...")
app = Flask(__name__)


def load_embedders(model_id_list):
    current_embedders = {
        os.environ['INTENT_CLASSIFIER_MODEL']: default_embedders['ru'],
        os.environ['INTENT_CLASSIFIER_ENG_MODEL']: default_embedders['en']
    }

    for model_id in model_id_list:
        print("Load embedder for model {} ...".format(model_id))
        model_path = os.path.join(
            MODEL_STORAGE, "model-{}.pickle".format(model_id)
        )

        with open(model_path, "rb") as handle:
            data = pickle.load(handle)

        embedder_path = data['embedder_path']
        language = data['language']

        if embedder_path not in current_embedders:
            embedder = Embedder(embedder_path, language)
            current_embedders[embedder_path] = embedder
        else:
            embedder = current_embedders[embedder_path]

        embedders[model_id] = embedder


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

    model_id = int(request.args.get("id"))

    try:
        embedder = embedders[model_id]
        classifier = IntentClassifier(embedder)
        classifier.load(model_id, MODEL_STORAGE)
    except (FileNotFoundError, KeyError):
        return jsonify({
            "message": "Model with id {} not found".format(model_id),
        })

    question = request.args.get("question")
    logging.info('predicting intent for question: {}'.format(question))

    if "top" in request.args:
        try:
            top = int(request.args.get("top"))
        except:
            return jsonify({
                "message": "Please specify `top` key as int value"
            })

        result_in_old_way = False
    else:
        # backward compatibility
        top = 1
        result_in_old_way = True

    model_answer = classifier.predict_top(question.strip(), top)

    output = []
    for answer, probability in model_answer:
        if classifier.label_to_accuracy_score:
            label = classifier.answer_to_label[answer]
            accuracy_score = classifier.label_to_accuracy_score[label]
        else:
            accuracy_score = None

        output.append({
            "answer": answer,
            "probability": probability,
            "threshold": classifier.threshold,
            "accuracy_score": accuracy_score,
        })

        logging.info('predicted intent: {}'.format(answer))
        logging.info('probability: {}'.format(probability))

    return jsonify(output[0]) if result_in_old_way else jsonify(output)


@app.route("/train", methods=["GET", "POST"])
def train():
    if (
        request.method not in ["GET", "POST"] or
        ("intent_json" not in request.args and
            "intent_json" not in request.form)
    ):
        return jsonify({
            "message": ("Please sent GET or POST query with `intent_json`,"
                        " `lang` keys"),
        })

    if 'lang' in request.args:
        language = request.args['lang']
    elif 'lang' in request.form:
        language = request.form['lang']
    else:
        language = 'ru'

    intentclf = IntentClassifier(
        default_embedders.get(language, 'ru')
    )
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
    intentclf.threshold_calc()
    intentclf.cross_val()

    new_model_id = intentclf.save(MODEL_STORAGE)
    load_embedders([new_model_id])
    logging.info('saved model with id {}'.format(new_model_id))

    return jsonify({
        "message": 'Created model with id {}'.format(new_model_id),
        "id": new_model_id,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Launch intent classifier API'
    )

    parser.add_argument(
        '-m', '--model_id_list', nargs='+', type=int
    )

    args = parser.parse_args()
    load_embedders(args.model_id_list)

    port = 3339 if os.path.isfile("_DEVELOP") else 3335
    app.run(host="0.0.0.0", port=port, debug=False)
