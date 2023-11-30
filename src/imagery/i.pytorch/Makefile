MODULE_TOPDIR = ..

SHELL_OUTPUT := $(shell python3 pytorchlib/dependency.py 2>&1)
ifeq ($(filter File pytorchlib/dependency.py,$(SHELL_OUTPUT)),)
    $(info $(SHELL_OUTPUT))
else
    $(error $(SHELL_OUTPUT))
endif

ALL_SUBDIRS := ${sort ${dir ${wildcard */.}}}
DEPRECATED_SUBDIRS := ${sort ${dir ${wildcard */DEPRECATED}}}
SUBDIRS := $(filter-out $(DEPRECATED_SUBDIRS), $(ALL_SUBDIRS))

include $(MODULE_TOPDIR)/include/Make/Dir.make

default: parsubdirs

install: installsubdirs
