#!/bin/bash
now=$(date +'%H-%M-%S-%Y-%m-%d')
python -u ./pipeline_orchestator_sequential.py > ${now}-sequential-pipeline.log
