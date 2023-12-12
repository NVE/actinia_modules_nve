MODULE_TOPDIR = ..

PGM = i.pytorch

SHELL_OUTPUT := $(shell python3 pytorchlib/dependency.py 2>&1)
ifeq ($(filter File pytorchlib/dependency.py,$(SHELL_OUTPUT)),)
    $(info $(SHELL_OUTPUT))
else
    $(error $(SHELL_OUTPUT))
endif

ALL_SUBDIRS := ${sort ${dir ${wildcard */.}}}
DEPRECATED_SUBDIRS := ${sort ${dir ${wildcard */DEPRECATED}}}
RM_SUBDIRS := bin/ docs/ scripts/
SUBDIRS_1 := $(filter-out $(DEPRECATED_SUBDIRS), $(ALL_SUBDIRS))
SUBDIRS := $(filter-out $(RM_SUBDIRS), $(SUBDIRS_1))

include $(MODULE_TOPDIR)/include/Make/Dir.make

default: parsubdirs

install: installsubdirs
	$(INSTALL_DATA) $(PGM).html $(INST_DIR)/docs/html/
