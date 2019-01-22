import unittest
import os
import shutil

from deepcubes.cubes import Embedder
from deepcubes.models import VeraLiveDialog


class TestVeraLiveDialog(unittest.TestCase):

    def setUp(self):
        emb_path = 'tests/data/test_embeds.kv'
        self.embedder = Embedder(emb_path)
        self.data_dir = 'tests/data'

        self.generic_data_path = 'tests/data/generic.txt'

        self.config = {
            "embedder_mode": "test",
            "tokenizer_mode": "lem",
            "labels_settings": [
                {
                    "label": "hello",
                    "patterns": ["привет", ".*привет.*"],
                    "intent_phrases": ["привет", "хай"]
                },
                {
                    "label": "bye-bye",
                    "patterns": ["пока", "пока-пока.*"],
                    "intent_phrases": ["пока", "прощай"]
                },
                {
                    "label": "rep",
                    "generics": ["repeat"]
                }],
            "not_understand_label": "not_understand"
        }
        self.config['lang'] = 'test'
        with open('new_vra_test.conf', 'w') as out:
            import json
            out.write(json.dumps(self.config))

    def test_vera_dialog(self):
        vera = VeraLiveDialog(self.embedder, self.generic_data_path)

        vera.train(self.config)

        # TODO: near equal checking
        self.assertEqual(
            vera("привет"),
            [("hello", 1), ("not_understand", 0.6),
             ("bye-bye", 0.5), ("rep", 0)]
        )

        # TODO: near equal checking
        self.assertEqual(
            vera("приветик"),
            [("hello", 1), ("not_understand", 0.6),
             ("bye-bye", 0.5), ("rep", 0)]
        )

        self.assertEqual(
            vera("прив пока-пока"),
            [("not_understand", 0.6), ("bye-bye", 0.5),
             ("hello", 0.5), ("rep", 0)]
        )

        self.assertEqual(
            vera("пока-пока привет"),
            [("bye-bye", 1), ("hello", 1),
             ("not_understand", 0.6), ("rep", 0)]
        )

        self.assertEqual(
            vera("повтори"),
            [("rep", 1), ("not_understand", 0.6),
             ("bye-bye", 0.5), ("hello", 0.5)]
        )

        self.assertEqual(
            vera("повтори", ["rep", "hello"]),
            [("rep", 1), ("not_understand", 0.6), ("hello", 0.5)]
        )

    def test_live_dialog_model_loading(self):
        vera = VeraLiveDialog(self.embedder, self.generic_data_path)
        vera.train(self.config)

        name = 'live_dialog.cube'
        model_id = 1
        model_path = os.path.join(self.data_dir, str(model_id))

        new_path = vera.save(name=name, path=model_path)
        new_vera = VeraLiveDialog.load(path=new_path)

        self.assertEqual(vera.config, new_vera.config)

        self.assertEqual(
            vera.intent_classifier.tokenizer.mode,
            new_vera.intent_classifier.tokenizer.mode
        )

        self.assertEqual(
            vera.intent_classifier.embedder.path,
            new_vera.intent_classifier.embedder.path
        )

        self.assertEqual(
            vera.intent_classifier.log_reg_classifier.clf.coef_.tolist(),
            new_vera.intent_classifier.log_reg_classifier.clf.coef_.tolist()
        )

        self.assertEqual(
            vera.generics['yes'].labels,
            new_vera.generics['yes'].labels
        )

        self.assertEqual(
            vera.generics['no'].labels,
            new_vera.generics['no'].labels
        )

        self.assertEqual(
            vera.generics['repeat'].labels,
            new_vera.generics['repeat'].labels
        )

        shutil.rmtree(os.path.join(self.data_dir, str(model_id)))
