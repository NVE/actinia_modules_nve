"""Test g.remove.path

(C) 2024 by NVE, Stefan Blumentrath and the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""

from pathlib import Path

import grass.script as gs
from grass.gunittest.case import TestCase
from grass.gunittest.gmodules import SimpleModule


class TestGRemovePath(TestCase):
    """Test case for removing files and directories"""

    @classmethod
    def setUpClass(cls):
        """Initiate the working environment"""
        cls.tempdir = Path(gs.tempdir())
        (cls.tempdir / "dir_1").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "dir_2").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "dir_3").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "dir_4").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "dir_5").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "dir_4" / "dir_1").mkdir(parents=True, exist_ok=True)
        (cls.tempdir / "file_1").write_text("", encoding="UTF8")
        (cls.tempdir / "file_2").write_text("", encoding="UTF8")
        (cls.tempdir / "file_3").write_text("", encoding="UTF8")
        (cls.tempdir / "dir_3" / "file_1").write_text("", encoding="UTF8")
        (cls.tempdir / "dir_4" / "file_1").write_text("", encoding="UTF8")

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary data"""
        gs.utils.try_rmdir(cls.tempdir)

    def test_g_remove_path_wildcard_no_removal(self):
        """Test file removal dry-run with wildcard"""
        tmp_files_before = list(self.tempdir.glob("*"))
        tmp_files_before.sort()
        # Check that g.remove.path runs successfully
        g_remove_list = SimpleModule(
            "g.remove.path",
            path=f"{self.tempdir}/*",
        ).run()

        tmp_files_after = list(self.tempdir.glob("*"))
        tmp_files_after.sort()
        # Check that no files are removed
        self.assertTrue(
            [str(tmp_file) for tmp_file in tmp_files_before]
            == [str(tmp_file) for tmp_file in tmp_files_after]
        )
        self.assertTrue(
            all(
                str(tmp_file) in g_remove_list.outputs.stderr
                for tmp_file in tmp_files_before
            )
        )

    def test_g_remove_path_wildcard_no_dir_removal(self):
        """Test file removal dry-run with wildcard"""
        tmp_files_before = list(self.tempdir.glob("dir*"))
        tmp_files_before.sort()
        # Check that g.remove.path runs successfully
        g_remove_list = SimpleModule(
            "g.remove.path", path=f"{self.tempdir}/dir*", flags="f"
        ).run()

        tmp_files_after = list(self.tempdir.glob("dir*"))
        tmp_files_after.sort()
        # Check that no files are removed
        self.assertTrue(
            [str(tmp_file) for tmp_file in tmp_files_before]
            == [str(tmp_file) for tmp_file in tmp_files_after]
        )
        self.assertTrue(
            all(
                str(tmp_file) in g_remove_list.outputs.stderr
                for tmp_file in tmp_files_before
            )
        )

    def test_g_remove_path_wildcard_with_removal(self):
        """Test file removal with wildcard"""
        # Check that g.remove.path runs successfully
        SimpleModule(
            "g.remove.path",
            path=f"{self.tempdir}/dir_3/*",
            flags="rf",
        ).run()

        # Check that no files are removed
        self.assertTrue(len(list((self.tempdir / "dir_3").glob("*"))) == 0)

    def test_g_remove_path_no_dir_removal(self):
        """Test file removal with wildcard"""
        # Check that g.remove.path runs successfully
        g_remove_list = SimpleModule(
            "g.remove.path",
            path=f"{self.tempdir}/dir_4/*",
            flags="f",
        ).run()

        self.assertTrue("WARNING" in g_remove_list.outputs.stderr)
        # Check that no dirs are removed
        self.assertTrue(len(list((self.tempdir / "dir_4").glob("*"))) == 1)

    def test_g_remove_path_single_dir_removal(self):
        """Test removal of single dir"""
        # Check that g.remove.path runs successfully
        g_remove_list = SimpleModule(
            "g.remove.path",
            path=f"{self.tempdir}/dir_5",
            flags="rf",
        ).run()
        print(g_remove_list.outputs.stderr)
        # Check that the dir is removed
        self.assertFalse((self.tempdir / "dir_5").exists())
        self.assertTrue("WARNING" not in g_remove_list.outputs.stderr)


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
