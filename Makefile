# set default shell
SHELL := $(shell which bash)
FOLDER=$$(pwd)
# default shell options
.SHELLFLAGS = -c
NO_COLOR=\\e[39m
OK_COLOR=\\e[32m
ERROR_COLOR=\\e[31m
WARN_COLOR=\\e[33m
.SILENT: ;
default: help;   # default target

IMAGE_NAME=dreamquark/notebook-runner:latest

# Make function
# https://stackoverflow.com/questions/10858261/abort-makefile-if-variable-not-set

# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined var $1$(if $2, ($2)). Please use $1=value))

pull:
	echo "Pulling Dockerfile ${IMAGE_NAME}"
	docker pull ${IMAGE_NAME}
.PHONY: pull

init: pull
	echo "Changing current folder rights"
	sudo chmod 777 . -R
	echo "Using project name $$(basename ${FOLDER})"
	make _run cmd="bash -cl 'init_notebook $$(basename ${FOLDER})'"
.PHONY: init

shell:
	@:$(call check_defined, port, Jupyter port)
	echo "Opening Poetry shell on running notebook on port=$$port"
	docker exec -it $$(docker ps --filter ancestor=${IMAGE_NAME} --filter expose=$$port -q) bash -lc "poetry_shell"
.PHONY: shell


logs:
	@:$(call check_defined, port, Jupyter port)
	echo "Getting notebook logs on port=$$port"
	docker logs $$( docker ps --filter ancestor=${IMAGE_NAME} --filter expose=$$port -q) -f
.PHONY: logs

_run_notebook:
	@:$(call check_defined, port, Jupyter port)
	echo "Running notebook on port=$$port"
	docker run --rm -d -it --network=braindocker_brain -v ${FOLDER}:/work -w /work -p $$port:$$port -e "JUPYTER_PORT=$$port" ${IMAGE_NAME} ${cmd}
.PHONY: _run

stop:
	@:$(call check_defined, port, Jupyter port)
	echo "Stopping container $$(docker ps --filter ancestor=${IMAGE_NAME} --filter expose=$$port -q) ..."
	docker stop $$(docker ps --filter ancestor=${IMAGE_NAME} --filter expose=$$port -q)
.PHONY: stop

_run:
	docker run --rm -it -v ${FOLDER}:/work -w /work ${IMAGE_NAME} ${cmd}
.PHONY: _run

root_bash:
	@:$(call check_defined, port, Jupyter port)
	docker exec -it --user root $$(docker ps --filter ancestor=${IMAGE_NAME} --filter expose=$$port -q) bash
.PHONY: root_bash


run:
	@:$(call check_defined, port, Jupyter port)
	make _run_notebook
.PHONY: run

help:
	echo -e "make [ACTION] <OPTIONAL_ARGS>"
	echo
	echo -e "This image uses Poetry for dependency management (https://poetry.eustace.io/)"
	echo
	echo -e "Default port for Jupyter notebook is 8888"
	echo
	echo -e "$(UDLINE_TEXT)ACTIONS$(NORMAL_TEXT):"
	echo -e "- $(BOLD_TEXT)init$(NORMAL_TEXT): create pyproject.toml interactive and install virtual env"
	echo -e "- $(BOLD_TEXT)run$(NORMAL_TEXT) port=<port>: run the Jupyter notebook on the given port"
	echo -e "- $(BOLD_TEXT)stop$(NORMAL_TEXT) port=<port>: stop the running notebook on this port"
	echo -e "- $(BOLD_TEXT)logs$(NORMAL_TEXT) port=<port>: show and tail the logs of the notebooks"
	echo -e "- $(BOLD_TEXT)shell$(NORMAL_TEXT) port=<port>: open a poetry shell"
.PHONY: help
