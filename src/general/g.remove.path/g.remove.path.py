#! /usr/bin/python3
"""
MODULE:    g.remove.path
AUTHOR(S): Stefan Blumentrath
PURPOSE:   Remove temporary files or directories
COPYRIGHT: (C) 2024 by Stefan Blumentrath and the GRASS Development Team

This program is free software under the GNU General
Public License (>=v2). Read the file COPYING that
comes with GRASS for details.
"""

# %module
# % description: Remove temporary files or directories
# % keyword: general
# % keyword: remove
# % keyword: file
# % keyword: directory
# % keyword: cleanup
# %end

# %option
# % key: path
# % description: Path to the file or directory to remove
# % required: yes
# %end

# %flag
# % key: f
# % label: Force removal
# % description: Force removal of files or directories
# %end

# %flag
# % key: r
# % label: Remove directories recursively
# % description: Remove directories recursively
# %end

# ruff: noqa: PTH207

import shutil
import sys
from glob import glob
from pathlib import Path

import grass.script as gs


def main():
    """Do the main work"""
    options, flags = gs.parser()
    paths_to_remove = glob(options["path"])
    if not paths_to_remove:
        gs.warning(_("Nothing found to remove with <{}>.").format(options["path"]))

    if flags["f"]:
        gs.info(_("Removing the following files and directories:"))
        for user_path in paths_to_remove:
            user_path = Path(user_path)
            if user_path.is_symlink() or user_path.is_file():
                try:
                    user_path.unlink()
                except Exception:
                    gs.warning(
                        _("Could not remove file or symlink <{}>").format(user_path)
                    )
            elif user_path.is_dir():
                if flags["r"]:
                    try:
                        shutil.rmtree(str(user_path))
                    except Exception:
                        gs.warning(
                            _("Could not remove directory <{}>").format(user_path)
                        )
                else:
                    gs.warning(
                        _(
                            "Cannot remove <{}>. It is a directory. Use the r-flag to remove it."
                        ).format(user_path)
                    )
    else:
        gs.info(_("Set to remove the following files and directories:"))
        gs.info(_("Use the f-flag to actually remove them."))
        gs.info(_("\n".join(paths_to_remove)))


if __name__ == "__main__":
    sys.exit(main())
