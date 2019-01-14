from flask import Flask, request, jsonify
from deepcubes.cubes import Embedder
import os

print("Prepare app...")
app = Flask(__name__)

print("Load embedders...")
emb_dict = {
    'rus': Embedder(),
    'eng': Embedder(),
    'test': Embedder(),
}
emb_dict['rus'].train(os.environ['INTENT_CLASSIFIER_MODEL']),
emb_dict['eng'].train(os.environ['INTENT_CLASSIFIER_ENG_MODEL']),
emb_dict['test'].train('tests/data/test_embeds.kv'),


@app.route("/get_vector", methods=["GET"])
def get_vector():
    if (
        request.method != "GET" or
        "tokens" not in request.args
    ):
        tokens = list()
    else:
        tokens = request.args.getlist("tokens")
    if (
        "tag" not in request.args
    ):
        return jsonify({
            "message": "Please sent GET query with `tokens` and `tag` keys",
        })

    tag = request.args.get("tag")

    if tag not in emb_dict:
        return jsonify({
            "message": "Bad `tag` key",
        })
    embedder = emb_dict[tag]
    vector = list(embedder(tokens))
    return jsonify({'vector': vector})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3349, debug=False)
