from flask import Flask, request, jsonify
import configparser
import logging
import sys
import os

from deepcubes.embedders import LocalEmbedder

logger = logging.getLogger("EmbedderService")
logger.setLevel(logging.INFO)

# create the logging file handler
handler = logging.FileHandler("scripts/logs/embedder_service.log")
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Started Embedder Server...")

# TODO: move from os.environ to script <args>
if 'SERVICE_CONF' in os.environ:
    config_file_path = os.environ['SERVICE_CONF']
else:
    logger.warning('Config file not found. Test config is used...')
    config_file_path = 'tests/data/embedder_service/embedder_service.conf'

config_parser = configparser.ConfigParser()
config_parser.read(config_file_path)

logger.info("Load embedders: {} and paths: {} ...".format(
    list(config_parser['embedders'].keys()),
    list(config_parser['embedders'].values())
))

embedders_dict = {name: LocalEmbedder(path)
                  for name, path in config_parser['embedders'].items()}

logger.info("Prepare Flask app...")

app = Flask(__name__)


@app.route("/<name>", methods=["GET"])
def get_vector(name):
    try:
        if name not in embedders_dict:
            logger.error("Attempt to use wrong embedder : {}".format(name))
            return jsonify({
                "message": "`{}` doesn't exists".format(name)
            })

        logger.info("Received embedder name: {}".format(name))
        embedder = embedders_dict[name]

        if "tokens" not in request.args:
            logger.error("Received request without tokens")
            return jsonify({
                "message": "Received request without `tokens`"
            })
        else:
            tokens = request.args.getlist("tokens")
            logger.info("Received tokens: {}".format(tokens))

        vector = list(embedder(tokens))

        logger.info("Sending vector...")
        return jsonify({'vector': vector})

    except Exception:
        _, exc_obj, exc_tb = sys.exc_info()
        error_line = "line {}, {}".format(exc_tb.tb_lineno, exc_obj)
        logger.error(error_line)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3343, debug=False)
