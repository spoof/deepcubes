import unittest
import numpy as np

import scripts.embedder_service as service
from deepcubes.cubes import Tokenizer


class VeraLiveDialogServiceTest(unittest.TestCase):

    def setUp(self):
        service.app.testing = True
        self.service = service.app.test_client()

    def test_get_requests(self):
        response = self.service.get('/test')
        self.assertEqual(response.status_code, 200)

        tokenizer = Tokenizer()
        tokenizer.train('lem')

        tokens = tokenizer('Робот Вера')
        response = self.service.get('/test', query_string={"tokens": tokens})

        generated_vector = response.get_json()["vector"]

        np.testing.assert_almost_equal(
            sum(generated_vector),
            -0.030646920857179794,
            1
        )
