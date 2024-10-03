"""Test t.rast.copytree

(C) 2024 by NVE, Stefan Blumentrath and the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""

# ruff: noqa: RUF012

from pathlib import Path

import grass.script as gs
from grass.gunittest.case import TestCase


class TestTRastCopytree(TestCase):
    """Test case for moving files from STRDS to directory trees"""

    default_region = {"s": 0, "n": 80, "w": 0, "e": 120, "b": 0, "t": 50}

    @classmethod
    def setUpClass(cls):
        """Initiate the temporal GIS and set the region"""
        cls.use_temp_region()
        cls.tempdir = Path(gs.tempdir())
        cls.tempdir_target = Path(gs.tempdir())
        cls.runModule("g.region", **cls.default_region, res=1, res3=1)

        cls.runModule(
            "r.external.out",
            format="GTiff",
            directory=str(cls.tempdir),
            extension="tif",
        )
        for rmap_idx in range(1, 4):
            for prefix in ("a", "b", "c"):
                cls.runModule(
                    "r.mapcalc",
                    expression=f"{prefix}_{rmap_idx} = {rmap_idx}00",
                    overwrite=True,
                )
                cls.runModule(
                    "r.support", map=f"{prefix}_{rmap_idx}", semantic_label=prefix
                )

        cls.runModule(
            "t.create",
            type="strds",
            temporaltype="absolute",
            output="A",
            title="A test",
            description="A test",
            overwrite=True,
        )
        cls.runModule(
            "t.register",
            flags="i",
            type="raster",
            input="A",
            maps="a_1,a_2,a_3",
            start="2001-01-01",
            increment="3 months",
            overwrite=True,
        )

        cls.runModule(
            "t.create",
            type="strds",
            temporaltype="absolute",
            output="B",
            title="B test",
            description="B test",
            overwrite=True,
        )
        cls.runModule(
            "t.register",
            flags="i",
            type="raster",
            input="B",
            maps="b_1,b_2,b_3",
            start="2001-01-01",
            increment="1 day",
            overwrite=True,
        )

        cls.runModule(
            "t.create",
            type="strds",
            temporaltype="absolute",
            output="C",
            title="C test",
            description="C test",
            overwrite=True,
        )
        cls.runModule(
            "t.register",
            flags="i",
            type="raster",
            input="C",
            maps="c_1,c_2,c_3",
            start="2001-01-01",
            increment="1 day",
            overwrite=True,
        )

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region"""
        cls.del_temp_region()
        cls.runModule("t.remove", flags="df", type="strds", inputs="A")
        cls.runModule("r.external.out", flags="r")
        gs.utils.try_rmdir(str(cls.tempdir_target))
        gs.utils.try_rmdir(str(cls.tempdir))

    def test_t_rast_copytree_move(self):
        """Test moving files into directory tree"""
        # Check that t.rast.copytree runs successfully
        self.assertModule(
            "t.rast.copytree",
            flags="m",
            input="A",
            temporal_tree="%Y/%m",
            output_directory=str(self.tempdir_target),
            nprocs="2",
        )
        self.assertFalse((self.tempdir / "a_1.tif").exists())
        self.assertFalse((self.tempdir / "a_2.tif").exists())
        self.assertFalse((self.tempdir / "a_3.tif").exists())

        self.assertFileExists(str(self.tempdir_target / "2001/01/a_1.tif"))
        self.assertFileExists(str(self.tempdir_target / "2001/04/a_2.tif"))
        self.assertFileExists(str(self.tempdir_target / "2001/07/a_3.tif"))

    def test_t_rast_copytree_copy_semantic_label(self):
        """Test moving files into directory tree"""
        # Check that t.rast.copytree runs successfully
        self.assertModule(
            "t.rast.copytree",
            flags="s",
            input="B",
            temporal_tree="%Y/%m/%d",
            output_directory=str(self.tempdir_target),
            nprocs="2",
        )

        self.assertFileExists(str(self.tempdir / "b_1.tif"))
        self.assertFileExists(str(self.tempdir / "b_2.tif"))
        self.assertFileExists(str(self.tempdir / "b_3.tif"))
        self.assertFileExists(str(self.tempdir_target / "b/2001/01/01/b_1.tif"))
        self.assertFileExists(str(self.tempdir_target / "b/2001/01/02/b_2.tif"))
        self.assertFileExists(str(self.tempdir_target / "b/2001/01/03/b_3.tif"))

    def test_t_rast_copytree_copy_overwrite(self):
        """Check that t.rast.copytree handles overwriting"""
        # Check that t.rast.copytree handles overwriting
        self.assertModule(
            "t.rast.copytree",
            input="C",
            temporal_tree="%Y/%m/%d",
            output_directory=str(self.tempdir_target),
            nprocs="2",
        )

        self.assertModuleFail(
            "t.rast.copytree",
            input="C",
            temporal_tree="%Y/%m/%d",
            output_directory=str(self.tempdir_target),
            nprocs="2",
        )

        self.assertModuleFail(
            "t.rast.copytree",
            flags="m",
            input="C",
            temporal_tree="%Y/%m/%d",
            output_directory=str(self.tempdir_target),
            nprocs="2",
        )

        self.assertModule(
            "t.rast.copytree",
            flags="om",
            input="C",
            temporal_tree="%Y/%m/%d",
            output_directory=str(self.tempdir_target),
            nprocs="2",
        )

        self.assertFalse((self.tempdir / "c_1.tif").exists())
        self.assertFalse((self.tempdir / "c_2.tif").exists())
        self.assertFalse((self.tempdir / "c_3.tif").exists())
        self.assertFileExists(str(self.tempdir_target / "2001/01/01/c_1.tif"))
        self.assertFileExists(str(self.tempdir_target / "2001/01/02/c_2.tif"))
        self.assertFileExists(str(self.tempdir_target / "2001/01/03/c_3.tif"))


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
