from flask import Flask, request, jsonify

from intentclf.models import Embedder
from intentclf.models import IntentClassifier


print("Load embedder...")
embedder = Embedder("/mnt/disk/models/glove-hh-embeds.kv")


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

    return jsonify({
        "answer": classifier.predict(request.args.get("question"))
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3334, debug=False)
