from flask import Flask, request, jsonify
import logging
import sys

from deepcubes.models import Sentiment

if len(sys.argv) < 2:
    print("Format: <path_to_model>")
    sys.exit()

logger = logging.getLogger("SentimentService")
logger.setLevel(logging.INFO)

# create the logging file handler
handler = logging.FileHandler("scripts/logs/sentiment_service.log")
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Started Sentiment Server...")

try:
    model = Sentiment.load(sys.argv[1])
except Exception:
    print("Error while loading the model")
    _, exc_obj, exc_tb = sys.exc_info()
    logger.error("line {}, {}".format(exc_tb.tb_lineno, exc_obj))
    sys.exit()

logger.info("Sentiment model has been loaded.")
logger.info("Prepare Flask app...")

app = Flask(__name__)


@app.route("/sentiment", methods=["GET"])
def sentiment():
    try:
        if request.method != "GET" or "query" not in request.args:
            logger.error("Received invalid request")
            return jsonify({
                "message": "Sent GET query with `query` keys",
            })

        logger.info("Received {} `sentiment` request from {}".format(
            request.method, request.remote_addr
        ))

        query = request.args.get("query")
        positive_proba = float(model(query))

        return jsonify({'positive_proba': positive_proba})

    except Exception:
        _, exc_obj, exc_tb = sys.exc_info()
        logger.error("line {}, {}".format(exc_tb.tb_lineno, exc_obj))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3342, debug=False)
