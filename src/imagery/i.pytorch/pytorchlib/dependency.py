#!/usr/bin/env python3

"""
MODULE:       dependency

AUTHOR(S):    Matej Krejci <matejkrejci gmail.com> (GSoC 2014),
              Tomas Zigo <tomas.zigo slovanet.sk>

PURPOSE:      Check i.pytorch py lib dependencies
              Mostly copied from wx.metadata

COPYRIGHT:    (C) 2020-2023 by Matej Krejci, Tomas Zigo,
              Stefan Blumentrath and the GRASS Development Team

              This program is free software under the GNU General
              Public License (>=v2). Read the file COPYING that
              comes with GRASS for details.
"""

import importlib
import sys

URL = "https://grasswiki.osgeo.org/wiki/ISO/INSPIRE_Metadata_Support"

MODULES = {
    "torch": {
        "check_version": False,
    },
    "numpy": {
        "check_version": False,
    },
}

INSTALLED_VERSION_MESSAGE = "Installed version of {} library is <{}>."
REQ_VERSION_MESSAGE = (
    "{name} {version} is required. check requirements on the manual page <{url}>."
)


def check_dependencies(module_name, check_version=False):
    """Check if py module is installed

    :param str module_name: py module name
    :param bool check_version: check py module version

    :return

    bool True: if py module is installed

    None: if py module is missing
    """

    module_cfg = MODULES[module_name]
    try:
        module = importlib.import_module(module_name)
        if module_cfg["check_version"]:
            message = "{inst_ver} {req_ver}".format(
                inst_ver=INSTALLED_VERSION_MESSAGE.format(
                    module_name,
                    module.__version__,
                ),
                req_ver=REQ_VERSION_MESSAGE.format(
                    name=module_name,
                    version=module_cfg["version"],
                    url=URL,
                ),
            )

            for index, package in enumerate(module_cfg["package"]):
                _package = importlib.import_module(package)

                if module_cfg.get("method"):
                    for method in module_cfg.get("method")[index]:
                        if not hasattr(_package, method):
                            sys.stderr.write(message)

                elif module_cfg.get("module"):
                    for module in module_cfg.get("module")[index]:
                        try:
                            importlib.import_module(module)
                        except ModuleNotFoundError:
                            sys.stderr.write(message)

        return True
    except ModuleNotFoundError:
        message = "{name} {text} <{url}>.\n".format(
            name=module_name,
            text="library is missing. Check requirements on the manual page",
            url=URL,
        )
        sys.stderr.write(message)


def main():
    for module in MODULES:
        if check_dependencies(module_name=module):
            print(f"{module} is installed.")


if __name__ == "__main__":
    sys.exit(main())
