from flask import Flask, request, jsonify
import os
import datetime

from intentclf.models import Embedder
from intentclf.models import IntentClassifier

print("Load embedder...")
embedder = Embedder(os.environ['INTENT_CLASSIFIER_MODEL'])

print("Prepare app...")
app = Flask(__name__)


@app.route("/answer", methods=["GET"])
def answer():
    if (request.method != "GET" or
            "id" not in request.args or
            "question" not in request.args):
        return jsonify({
            "message": "Please sent GET query with `id` and `question` keys"
        })

    classifier = IntentClassifier(embedder)
    classifier.load(
        "scripts/models/model-{}.pickle".format(request.args.get("id"))
    )

    answer = classifier.predict(request.args.get("question"))

    with open("scripts/logs/log.txt", "a") as log_file:
        print("{}: {}\t{}".format(
            datetime.datetime.now(),
            request.args.get("question"),
            answer
        ), file=log_file)

    return jsonify({
        "answer": answer
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3335, debug=False)
