import unittest

from ar_opd import __version__


class PackageTest(unittest.TestCase):
    def test_package_version_is_available(self) -> None:
        self.assertEqual(__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
