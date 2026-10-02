"""Release-identity checks for the public SURPASS package."""

import unittest

import surpass


class TestReleaseIdentity(unittest.TestCase):
    def test_public_version(self):
        self.assertEqual(surpass.__version__, "1.0.0")

    def test_material_independent_presets(self):
        expected = {
            "Si": "silicon",
            "Ge": "germanium",
            "InP": "indium phosphide",
            "GaAs": "gallium arsenide",
        }
        for lookup, canonical_name in expected.items():
            self.assertEqual(surpass.get_material(lookup).name, canonical_name)


if __name__ == "__main__":
    unittest.main()
