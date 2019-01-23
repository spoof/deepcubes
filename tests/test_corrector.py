import unittest
from deepcubes.cubes import Corrector


class TestCorrector(unittest.TestCase):

    def test_corrector(self):
        corrector = Corrector()

        corrector.train(["привет", "пока", "ура", "нет"], 1)

        self.assertEqual(
            corrector(["привет", "пока"]),
            ["привет", "пока"]
        )

        self.assertEqual(
            corrector(["привут", "пока"]),
            ["привет", "пока"]
        )

        self.assertEqual(
            corrector(["xaxa", "поку"]),
            ["xaxa", "пока"]
        )

        corrector.train(["привет", "пока", "ура", "нет"], 0)

        self.assertEqual(
            corrector(["xaxa", "поку", ""]),
            ["xaxa", "поку", ""]
        )
