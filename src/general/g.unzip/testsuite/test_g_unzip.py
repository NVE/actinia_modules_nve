"""Test i.satskred

(C) 2023 by NVE, Stefan Blumentrath and the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""

from pathlib import Path
from zipfile import ZipFile, ZipInfo

import grass.script as gs

from grass.gunittest.case import TestCase


class TestGUnzipParallel(TestCase):
    """Test case for parallel unzipping"""

    @classmethod
    def setUpClass(cls):
        """Initiate the working environment"""
        cls.tempdir = Path(gs.tempdir())
        for file_number in range(4):
            with ZipFile(cls.tempdir / f"testfile_{file_number}.zip", "w") as test_zip:
                zipinfo_dir = ZipInfo("./testdir/")
                test_zip.writestr(zipinfo_dir, "")
                test_zip.writestr(
                    f"./testdir/testfile_{file_number}.txt", "This is test content!"
                )

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary data"""
        gs.utils.try_rmdir(cls.tempdir)

    def test_g_unzip_no_out_dir(self):
        """Test unzipping to output directory while
        removing zip files in parallel"""
        # Check that zip-files are created
        self.assertTrue(len(list(self.tempdir.glob("testfile_*.zip"))) == 4)
        # Check that g.unzip runs successfully
        self.assertModule(
            "g.unzip",
            input=str(self.tempdir),
            nprocs=2,
            verbose=True,
        )
        # Check that unzipped-files exist in the file system
        self.assertTrue(
            len(list((self.tempdir / "testdir").glob("testfile_*.txt"))) == 4
        )

    def test_g_unzip_with_out_dir(self):
        """Test unzipping to output directory while
        removing zip files in parallel"""
        # Check that zip-files are created
        self.assertTrue(len(list(self.tempdir.glob("testfile_*.zip"))) == 4)
        # Check that g.unzip runs successfully
        self.assertModule(
            "g.unzip",
            input=str(self.tempdir),
            output=str(self.tempdir / "output_directory"),
            nprocs=2,
            flags="r",
            verbose=True,
        )
        # Check that zip-files are removed as requested
        self.assertTrue(len(list(self.tempdir.glob("testfile_*.zip"))) == 0)
        # Check that unzipped-files exist in the file system
        self.assertTrue(
            len(
                list(
                    (self.tempdir / "output_directory" / "testdir").glob(
                        "testfile_*.txt"
                    )
                )
            )
            == 4
        )


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
