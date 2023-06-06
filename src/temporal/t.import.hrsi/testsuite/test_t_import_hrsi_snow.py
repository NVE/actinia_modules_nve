"""Test t.rast.reclass

(C) 2022 by the GRASS GIS Development Team
This program is free software under the GNU General Public
License (>=v2). Read the file COPYING that comes with GRASS
for details.

:authors: Stefan Blumentrath
"""
import os

import grass.pygrass.modules as pymod
import grass.temporal as tgis
from grass.gunittest.case import TestCase
from grass.gunittest.gmodules import SimpleModule


class TestAggregationAbsolute(TestCase):
    @classmethod
    def setUpClass(cls):
        """Initiate the temporal GIS and set the region"""
        os.putenv("GRASS_OVERWRITE", "1")

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary region and data"""
        products = [
            "FractionalSnowCover",
            "GapfilledFractionalSnowCover",
            "PersistentSnowArea",
            "PersistentSnowArea_LAEA",
            "RiverandLakeIceExtent_S1",
            "RiverandLakeIceExtent_S2",
            "RiverandLakeIceExtent_S1_S2",
            "SARWetSnow",
            "WetDrySnow",
            "AggregatedRiverandLakeIceExtent",
        ]
        for product in products:
            try:
                cls.runModule("t.remove", flags="df", type="strds", inputs=product)
            except Exception:
                pass

    def test_import_WDS(self):
        """Test download and import of WetDrySnow with fast external data"""
        test_product = "WetDrySnow"
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
        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)

    def test_import_SWS(self):
        """Test download and import of SARWetSnow with fast external data"""
        test_product = "SARWetSnow"
        self.assertModule(
            "t.import.hrsi",
            flags="f",
            product_type=test_product,
            aoi="./aoi.geojson",
            start_time="2023-04-05T00:00:00",
            end_time="2023-04-05T23:50:59",
            nprocs=2,
            memory=2048,
            output=test_product,
        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)

    def test_import_FSC(self):
        """Test download and import of FractionalSnowCover with external data"""
        test_product = "FractionalSnowCover"
        self.assertModule(
            "t.import.hrsi",
            flags="l",
            product_type=test_product,
            aoi="./aoi.geojson",
            start_time="2023-04-01T00:00:00",
            end_time="2023-04-01T23:59:59",
            nprocs=2,
            memory=2048,
            output=test_product,
        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)

    def test_import_GFSC(self):
        """Test download and import of GapfilledFractionalSnowCover with external data"""
        test_product = "GapfilledFractionalSnowCover"
        self.assertModule(
            "t.import.hrsi",
            flags="l",
            product_type=test_product,
            aoi="./aoi.geojson",
            start_time="2023-04-04T00:00:00",
            end_time="2023-04-04T23:59:59",
            nprocs=2,
            memory=2048,
            output=test_product,
        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)

    def test_import_PSA(self):
        """Test download and import of PersistentSnowArea with external data"""
        test_product = "PersistentSnowArea"
        self.assertModule(
            "t.import.hrsi",
            flags="l",
            product_type=test_product,
            aoi="./aoi.geojson",
            start_time="2022-05-01T00:00:00",
            end_time="2022-09-30T23:59:59",
            nprocs=2,
            memory=2048,
            output=test_product,
        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)

    def test_import_PSA_LAEA(self):
        """Test download and import of PersistentSnowArea_LAEA"""
        test_product = "PersistentSnowArea_LAEA"
        self.assertModule(
            "t.import.hrsi",
            product_type=test_product,
            aoi="./aoi.geojson",
            start_time="2022-05-01T00:00:00",
            end_time="2022-09-30T23:59:59",
            nprocs=2,
            memory=2048,
            output=test_product,
        )
        info = SimpleModule(
            "t.info",
            flags="g",
            input=test_product,
        ).run()
        print(info.outputs.stdout)
        tlist = SimpleModule(
            "t.rast.list",
            input=test_product,
        ).run()
        print(tlist.outputs.stdout)


if __name__ == "__main__":
    from grass.gunittest.main import test

    test()
