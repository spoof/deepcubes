import json
import pandas as pd
import argparse
from gensim.models import Word2Vec
from deepcubes.cubes import Tokenizer

conf_path = 'tests/data/vera_test.config'
generic_path = 'tests/data/generic.txt'

with open(conf_path, 'r') as f:
    config = json.loads(f.read())

df = pd.read_csv(generic_path, sep='\t', names=['text', 'type'])

generic_sentences = list()
for value in df.text:
    generic_sentences.append(value)

config_sentences = list()
for _dict in config['labels_settings']:
    if 'intent_phrases' in _dict:
        for phrase in _dict['intent_phrases']:
            config_sentences.append(phrase)

sentences = generic_sentences + config_sentences
tokenizer = Tokenizer()


def prepare_embedder(mode, letter_limit):
    tokenizer.train(mode, letter_limit)
    tok_sentenses = list()

    for phrase in sentences:
        tok_sentenses.append(tokenizer(phrase))

    model = Word2Vec(tok_sentenses, min_count=1)
    model.wv.save('tests/data/test_embeds.kv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare embedder fo tests')
    parser.add_argument('-m', '--mode',
                        help='Mode of tokenizer', default='lem')
    parser.add_argument('-l', '--letter_limit',
                        help='Mode of tokenizer', type=int, default=0)
    args = parser.parse_args()

    prepare_embedder(args.mode, args.letter_limit)
