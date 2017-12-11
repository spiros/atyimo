#!/bin/bash

## FEDERAL UNIVERSITY OF BAHIA (UFBA)
## ATYIMOLAB (www.atyimolab.ufba.br)
## University College London
## Denaxas Lab (www.denaxaslab.org)

# File:           $atyimo_script.sh$
# Version:        $v1$
# Last changed:   $Date: 2017/12/04 12:00:00 $
# Purpose:        Basic execution flow for the AtyImo data linkage tool
# Author:         Robespierre Pita and Clicia Pinto and Marcos Barreto and Spiros Denaxas

# Usage:  ./atyimo_script.sh

# Comments:

# Preprocessing
spark-submit 01_preprocessing.py

# CreateBlockKey
spark-submit 02_createBlockKey.py

# WriteBlocks
# if blocking is set, this file should be executed after 04_encoding_blocking.py
spark-submit 03_writeBlocks.py

# Enconding
# if blocking is set, this file should be executed before 03_writeBlocks.py
spark-submit 04_encoding_blocking.py

# Correlation
#spark-submit 05_correlation_new_otimizado2.py
spark-submit 05_correlation.py

# DedupByKey
spark-submit 06_dedupByKey.py

# geraDataMart
spark-submit 07_geraDataMart.py
