#!/bin/bash

ENV_NAME="treeseg_dev"

conda create -n $ENV_NAME

conda env update -n $ENV_NAME -f environment.yaml
