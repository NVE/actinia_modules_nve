"""Test t.import.hrsi for Ice products

(C) 2023 by the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""
import os

import grass.script as gs

from grass.gunittest.case import TestCase
from grass.gunittest.gmodules import SimpleModule


class TestAggregationAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        """Initiate the temporal GIS and set the region"""
        os.putenv("GRASS_OVERWRITE", "1")
        cls.tempdir = gs.tempdir()

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region and data"""
        products = [
            "RiverandLakeIceExtent_S1",
            "RiverandLakeIceExtent_S2",
            "RiverandLakeIceExtent_S1_S2",
        ]
        for product in products:
            try:
                cls.runModule("t.remove", flags="df", type="strds", inputs=product)
            except Exception:
                pass
        gs.utils.try_rmdir(cls.tempdir)

    def test_import_RILE_S1_S2(self):
        """Test download and import of RiverandLakeIceExtent_S1_S2 with fast external data"""
        test_product = "RiverandLakeIceExtent_S1_S2"
        self.assertModule(
            "t.import.hrsi",
            flags="f",
            product_type=test_product,
            aoi="./aoi.geojson",
            start_time="2023-04-05T00:00:00",
            end_time="2023-04-05T23:59:59",
            nprocs=2,
            memory=2048,
            output=test_product,
            output_directory=self.tempdir,
        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)

    def test_import_RILE_S2(self):
        """Test download and import of RiverandLakeIceExtent_S2 with fast external data"""
        test_product = "RiverandLakeIceExtent_S2"
        self.assertModule(
            "t.import.hrsi",
            flags="f",
            product_type=test_product,
            aoi="./aoi.geojson",
            start_time="2023-04-01T00:00:00",
            end_time="2023-04-01T23:59:59",
            nprocs=2,
            memory=2048,
            output=test_product,
            output_directory=self.tempdir,
        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)

    def test_import_RILE_S1(self):
        """Test download and import of RiverandLakeIceExtent_S1 with external data"""
        test_product = "RiverandLakeIceExtent_S1"
        self.assertModule(
            "t.import.hrsi",
            flags="l",
            product_type=test_product,
            aoi="./aoi.geojson",
            start_time="2023-04-05T00:00:00",
            end_time="2023-04-05T23:59:59",
            nprocs=2,
            memory=2048,
            output=test_product,
            output_directory=self.tempdir,
        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
