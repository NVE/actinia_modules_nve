"""Test g.transfer

(C) 2024 by NVE, Stefan Blumentrath and the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""

from pathlib import Path

import grass.script as gs
from grass.gunittest.case import TestCase


class TestGTransfer(TestCase):
    """Test case for removing files and directories"""

    @classmethod
    def setUpClass(cls):
        """Initiate the working environment"""
        cls.tempdir = Path(gs.tempdir())
        cls.tempdir_target = Path(gs.tempdir())
        (cls.tempdir / "dir_1").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "dir_2").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "dir_3").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "dir_4").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "dir_4" / "dir_1").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "file_1").write_text("Test", encoding="UTF8")
        (cls.tempdir / "file_2").write_text("Test", encoding="UTF8")
        (cls.tempdir / "file_3").write_text("Test", encoding="UTF8")
        (cls.tempdir / "dir_3" / "file_1").write_text("Test", encoding="UTF8")
        (cls.tempdir / "dir_4" / "file_1").write_text("Test", encoding="UTF8")

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary data"""
        gs.utils.try_rmdir(cls.tempdir)
        gs.utils.try_rmdir(cls.tempdir_target)

    def test_g_transfer_no_removal(self):
        """Test copying files and directories"""

        # Check that g.transfer runs successfully
        self.assertModule(
            "g.transfer",
            source=f"{self.tempdir}/dir_4",
            target=f"{self.tempdir_target}",
        )

        # Check that no files are removed
        self.assertTrue((self.tempdir / "dir_4").is_dir())
        self.assertTrue((self.tempdir / "dir_4" / "dir_1").is_dir())
        self.assertFileExists(str(self.tempdir / "dir_4" / "file_1"))
        # Check that files are copied
        self.assertTrue((self.tempdir_target / "dir_4").is_dir())
        self.assertTrue((self.tempdir_target / "dir_4" / "dir_1").is_dir())
        self.assertFileExists(str(self.tempdir_target / "dir_4" / "file_1"))

    def test_g_transfer_move(self):
        """Test moving files and directories"""

        # Check that g.transfer runs successfully
        self.assertModule(
            "g.transfer",
            flags="m",
            source=f"{self.tempdir}/dir_3",
            target=f"{self.tempdir_target}",
        )

        # Check that no files are removed
        self.assertFalse((self.tempdir / "dir_3").is_dir())
        self.assertFalse((self.tempdir / "dir_3" / "file_1").is_file())
        # Check that files are copied
        self.assertTrue((self.tempdir_target / "dir_3").is_dir())
        self.assertFileExists(str(self.tempdir_target / "dir_3" / "file_1"))

    def test_g_transfer_with_wildcard(self):
        """Test copying files with wildcard"""

        # Check that g.transfer runs successfully
        self.assertModule(
            "g.transfer",
            source=f"{self.tempdir}/file_*",
            target=f"{self.tempdir_target}",
            nprocs="2",
        )

        # Check that no files are removed
        self.assertTrue((self.tempdir / "file_1").is_file())
        self.assertTrue((self.tempdir / "file_2").is_file())
        self.assertTrue((self.tempdir / "file_3").is_file())
        # Check that files are copied
        self.assertTrue((self.tempdir / "file_1").is_file())
        self.assertTrue((self.tempdir / "file_2").is_file())
        self.assertTrue((self.tempdir / "file_3").is_file())


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
