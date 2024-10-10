#!/bin/bash

# Remove existing relevant directories in separate services
rm -rf prediction_api/model_registry/

# Copy the model directory into prediction_api
cp -r model_registry prediction_api/

