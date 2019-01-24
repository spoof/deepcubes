from flask import Flask, request, jsonify
import configparser
import logging
import json
import sys
import os

from deepcubes.cubes import Embedder

logger = logging.getLogger("EmbedderService")
logger.setLevel(logging.INFO)

# create the logging file handler
fh = logging.FileHandler("scripts/logs/emb_service.log")
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("Started Embedder Server...")

if 'SERVICE_CONF' in os.environ:
    config_file_path = os.environ['SERVICE_CONF']
else:
    logger.warning('Config file not found. Test config is used...')
    config_file_path = 'tests/data/test.conf'

config_parser = configparser.RawConfigParser()
config_parser.read(config_file_path)

logger.info("Load embedders with langs: {} and paths: {} ...".format(
    list(config_parser['embedder'].keys()),
    list(config_parser['embedder'].values())
))
emb_dict = {lang: Embedder(json.loads(path))
            for lang, path in config_parser['embedder'].items()}

logger.info("Prepare Flask app...")
app = Flask(__name__)


@app.route("/get_vector", methods=["GET"])
def get_vector():
    try:
        if request.method != "GET" or "mode" not in request.args:
            logger.error("Received invalid request")
            return jsonify({
                "message": "Sent GET query with `tokens` and `mode` keys",
            })

        logger.info("Received {} `predict` request from {}".format(
            request.method, request.remote_addr
        ))

        mode = request.args.get("mode")
        if mode not in emb_dict:
            logger.error("Received invalid `mode` : {}".format(mode))
            return jsonify({
                "message": "Bad `mode` key",
            })

        logger.info("Received embedder mode: {}".format(mode))

        if "tokens" not in request.args:
            logger.error("Received request without tokens")
            tokens = list()
        else:
            tokens = request.args.getlist("tokens")
            logger.info("Received tokens: {}".format(tokens))

        embedder = emb_dict[mode]
        vector = list(embedder(tokens))
        logger.info("Sending vector...")

        return jsonify({'vector': vector})

    except Exception:
        _, exc_obj, exc_tb = sys.exc_info()
        logger.error("line {}, {}".format(exc_tb.tb_lineno, exc_obj))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3349, debug=False)
