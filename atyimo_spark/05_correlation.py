#!/bin/env python
# coding: utf-8

## FEDERAL UNIVERSITY OF BAHIA (UFBA)
## ATYIMOLAB (www.atyimolab.ufba.br)
## University College London
## Denaxas Lab (www.denaxaslab.org)

# File:           $05_correlation.py$
# Version:        $v1$
# Last changed:   $Date: 2017/12/04 12:00:00$
# Purpose:        $Correlate records, generate candidate pairs, apply similarity index and recover pairs above the cutoff$
# Author:         Robespierre Pita and Clicia Pinto and Marcos Barreto and Spiros Denaxas

# Usage:  /path/to/python/05_correlation.py

# Comments:
# you need to compile cb.c to generate a dependent library (see cb.c for instructions)
# generates the file result.linkage under the folder linkage_results/
# Structure: dice ; index_larger_database ; index_smaller_database

from pyspark import SparkContext, SparkConf
from pyspark import SparkFiles
import csv
import time
import os
import os.path
import commands
import config
import config_static
from datetime import datetime
from pyspark.storagelevel import StorageLevel
import gc
import array
from ctypes import *
ini = time.time() # Initializing time couting

conf = config_static.co_conf
sc = SparkContext(conf=conf)

print "MAIN PROGRAM MESSAGE (correlation):		  Correlation starting..."

# You may use that
#spark.conf.set("spark.executor.extraJavaOptions", "-XX:+UseG1GC")

def set_variables():
	print "MAIN PROGRAM MESSAGE (correlation):		  In set_variables()"

	global default_folder
	default_folder = config.default_folder

	global dm_type4
	dm_type4 = config.dm_type4

	global directory_results
	directory_results = config_static.c_directory_results

	global directory_block_larger
	directory_block_larger = config_static.e_directory_block_larger

	global directory_block_smaller
	directory_block_smaller = config_static.e_directory_block_smaller

	global resultFileName
	resultFileName = config_static.c_file_result

	global resultFileNameRescue
	resultFileNameRescue = config_static.c_file_result_rescue

	global size_bloom_vector
	size_bloom_vector = config_static.c_size_bloom_vector

	global cutoff_result
	cutoff_result = config.c_cutoff_result

	global cutoff_result_rescue
	cutoff_result_rescue = config_static.c_cutoff_result_rescue

def create_path():
        print "MAIN PROGRAM MESSAGE (correlation):		  In create_path()"
	global directory_results
	directory_results = directory_results
	os.system("mkdir "+directory_results)

def compare2(linharecebida):
	_cb = CDLL('libcb.so') #change this to the path of "libcb.so" compiled file
	_cb.calculate_h.argtypes = (c_char_p, c_char_p, c_int)
	arr = array.array
	bloomSize = 4
	largestdice = 0
	estrutura = ""
	folder2 = ''.join(folder.value)
	f = open(directory_block_larger+folder2,'rb')
        lrSplit = str(linharecebida).split(";")
        lrIndex = lrSplit[0]
	lrUns = lrSplit[1]
        lrBMin = lrSplit[2]
	lrBMax = lrSplit[3]
	lrBloom = lrSplit[4:]
	for line in f:
		h = 0
		bc = line.replace("\n","")
		bcSplit = bc.split(";")
		bcIndex = bcSplit[0]
		bcUns = bcSplit[1]
        	bcBMin = bcSplit[2]
		bcBloom = bcSplit[4:]
		equals = 0
		for i in range(min(len(bcBloom), len(lrBloom))):
			if bcBloom[i] == lrBloom[i]:
				equals += 1
		if equals >= bloomSize - 1:
			if bcBloom == lrBloom:
				dice = 10000
			else:
				dice = 9999
			estrutura = str(dice) + ";" + bcIndex + ";" + lrIndex
			break
		else:
			if (lrBMin < bcUns < lrBMax) or bcBMin == 0:
				try:
					zipped = zip(bcBloom,lrBloom)
					vetor1 = "".join([x for x,y in zipped if len(y) > 0])
					vetor2 = "".join([y for x,y in zipped if len(x) > 0])
					h = _cb.calculate_h(c_char_p(vetor1), c_char_p(vetor2), c_int(len(vetor1)))
				except Exception:
					imenor = lrIndex
					imaior = bcIndex
					out = str(imaior) + ';' + str(imenor) + ';' + str(folder2)
					n = open('log_blocos_com_erro', 'a')
					n.write(out + "\n")
					h = 0
				dice = ((2 * float(h))/(float(bcUns) + float(lrUns)))*10000
				dice = round(dice,0)
			else:
				dice = 8900.0
		if dice > largestdice:
			largestdice = dice
			estrutura = str(dice) + ";" + bcIndex + ";" + lrIndex

	k = open(resultFileName, 'a')
	k.write(str(estrutura) + "\n")
	f.close()
	return estrutura

print "MAIN PROGRAM MESSAGE (correlation):		  Starting variables..."
set_variables()
create_path()

if (dm_type4):
	arquivosBlocos2SIH = os.listdir(directory_block_smaller)
	times = []
	for a in arquivosBlocos2SIH:
		if os.path.exists(directory_block_larger + a) and a != "tmp_":
			now = datetime.now()
			log = str(now) + " : Initializing block " + str(a)
			q = open('estimative_logs.txt', 'a')
			q.write(str(log) + "\n")
			p = time.time()
			if(config.e_status_blocking):
				var=sc.parallelize(a).cache().collect()
				folder=sc.broadcast(var)
				smaller = sc.textFile(directory_block_smaller + a)
				c = smaller.cache().count()
				if( c > 8):
					tot = 32
				else:
					tot = 4
				smaller = sc.textFile(directory_block_smaller + a, tot).cache()
			else:
				var=sc.parallelize(a).cache().collect()
				folder=sc.broadcast(var)
				smaller = sc.textFile(directory_block_smaller + "SINGLE_BLOCK.bloom", 1024)
				larger = sc.textFile(directory_block_larger + a, 2048)

			registrospareados = smaller.cache().map(compare2).collect()

			smaller.unpersist()

			k = open('/PATH/TO/completed_blocks', 'a')
			k.write(str(a) + "\n")
			final_t = time.time() - p
			times.append(final_t)
			n_avg = sum(times)/len(times)
			remaining = 100 - len(times)
			estimation = (n_avg * remaining)
			print "New estimation: " + str(estimation) + " with " +  str(remaining) + " jobs remaining"
			now = datetime.now()
			log = str(now) + " : Terminating block " + str(a)
			q.write(str(log) + "\n")
else:

	arquivosBlocos2SIH = os.listdir(directory_block_smaller)

	for a in arquivosBlocos2SIH:
		if os.path.exists(directory_block_larger + a):

			smaller = sc.textFile(directory_block_smaller + a, 1024)
			larger = sc.textFile(directory_block_larger + a, 256)

			var = smaller.cache().collect()
			varbc = sc.broadcast(var)

			registrospareados = larger.cache().map(compare).collect()

			smaller.unpersist()
			larger.unpersist()
fim = time.time()
approx_time = fim - ini
print "MAIN PROGRAM MESSAGE (correlation):		  Correlation completed in: " + str(approx_time)
