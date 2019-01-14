import unittest
import os
import shutil

from deepcubes.models import VeraLiveDialog


class TestVeraLiveDialog(unittest.TestCase):

    def setUp(self):
        # TODO: get path from environment
        self.emb_path = 'http://51.144.105.1:3349/get_vector'
        self.data_dir = 'tests/data'
        self.config = {
            "lang": "test",
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
                }]
        }

    def test_vera_dialog(self):
        vera = VeraLiveDialog(self.emb_path)

        vera.train(self.config)

        # TODO: near equal checking
        self.assertEqual(
            vera("привет"),
            [("hello", 1), ("bye-bye", 0.5)]
        )

        # TODO: near equal checking
        self.assertEqual(
            vera("приветик"),
            [("hello", 1), ("bye-bye", 0.5)]
        )

        self.assertEqual(
            vera("прив пока-пока"),
            [("hello", 0.5), ("bye-bye", 0.5)]
        )

        self.assertEqual(
            vera("пока-пока привет"),
            [("hello", 1), ("bye-bye", 1)]
        )

    def test_live_dialog_model_loading(self):
        vera = VeraLiveDialog(self.emb_path)
        vera.train(self.config)
        model_id = 1
        name = 'live_dialog.cube'
        new_path = vera.save(model_id=model_id, name=name, path=self.data_dir)
        new_vera = VeraLiveDialog.load(path=new_path)
        self.assertEqual(vera.config, new_vera.config)
        shutil.rmtree(os.path.join(self.data_dir, str(model_id)))
