#!/bin/env python
# coding: utf-8

## FEDERAL UNIVERSITY OF BAHIA (UFBA)
## ATYIMOLAB (www.atyimolab.ufba.br)
## University College London
## Denaxas Lab (www.denaxaslab.org)

# File:           $04_enconding_blocking.py$
# Version:        $v1$
# Last changed:   $Date: 2017/12/04 12:00:00$
# Purpose:        $Eliminate duplicate records given a specific key$
# Author:         Robespierre Pita and Clicia Pinto and Marcos Barreto and Spiros Denaxas

# Usage:  /path/to/python/04_enconding_blocking.py
# IMPORTANT: if blocking is set, you should execute 03_writeBlocks.py after this file

# Comments: generates files under the folders blocks/block_smaller and blocks/block_larger
# FILES.bloom -> write Bloom records from the input records
# SINGLE_BLOCK.bloom, if blocking is not set

from pyspark import SparkContext, SparkConf
from pyspark import SparkFiles
from unicodedata import normalize
from doctest import testmod
from operator import add
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import collections
import csv
import time
import hashlib
import os
import os.path
import commands
import config
import config_static
from datetime import datetime
ini = time.time() # Initializing time couting

conf = config_static.en_conf
sc = SparkContext(conf=conf)
print "MAIN PROGRAM MESSAGE (encoding):             Encoding starting..."
global y
y = 0

def set_variables():
    print "MAIN PROGRAM MESSAGE (encoding):             In set_variables()"

    global status_larger_base
    status_larger_base = config.status_larger_base
    global status_smaller_base
    status_smaller_base = config.status_smaller_base
    global default_folder
    default_folder = config.default_folder

    global status_blocking
    status_blocking = config.e_status_blocking

    global size_bloom_col_n
    size_bloom_col_n = config.e_size_bloom_col_n
    global size_bloom_col_mn
    size_bloom_col_mn = config.e_size_bloom_col_mn
    global size_bloom_col_bd
    size_bloom_col_bd = config.e_size_bloom_col_bd
    global size_bloom_col_mr
    size_bloom_col_mr = config.e_size_bloom_col_mr
    global size_bloom_col_g
    size_bloom_col_g = config.e_size_bloom_col_g

    global status_name
    status_name = config.e_status_name
    global status_birth_date
    status_birth_date = config.e_status_birth_date
    global status_gender
    status_gender = config.e_status_gender
    global status_mother_name
    status_mother_name = config.e_status_mother_name
    global status_municipality_residence
    status_municipality_residence = config.e_status_municipality_residence
    global status_state
    status_state = config.e_status_state

    global col_i
    col_i = config_static.e_col_i
    global col_n
    col_n = config_static.e_col_n
    global col_mn
    col_mn = config_static.e_col_mn
    global col_bd
    col_bd = config_static.e_col_bd
    global col_g
    col_g = config_static.e_col_g
    global col_mr
    col_mr = config_static.e_col_mr
    global col_st
    col_st = config_static.e_col_st

def set_variables_larger():
    print "MAIN PROGRAM MESSAGE (encoding):             In set_variables_larger()"

    global partitioning
    partitioning = config_static.larger_partitioning
    global input_file
    input_file = config_static.e_largest_input_file
    print "MAIN PROGRAM MESSAGE (encoding):             Input File: " +input_file
    global outputFolder
    outputFolder = directory_block_larger

def set_variables_smaller():
    print "MAIN PROGRAM MESSAGE (encoding):             In set_variables_smaller()"

    global partitioning
    partitioning = config_static.smaller_partitioning
    global input_file
    input_file = config_static.e_smaller_input_file
    print "MAIN PROGRAM MESSAGE (encoding):             Input FIle: " +input_file
    global outputFolder
    outputFolder = directory_block_smaller

def create_path():
    print "MAIN PROGRAM MESSAGE (encoding):             In create_path()"
    global directory_main
    directory_main = config_static.e_directory_blocks
    global directory_block_larger
    directory_block_larger = config_static.e_directory_block_larger
    global directory_block_smaller
    directory_block_smaller = config_static.e_directory_block_smaller

    if(status_blocking):
        os.system("mkdir "+directory_block_larger+"tmp_")
        os.system("mkdir "+directory_block_smaller+"tmp_")
        global keyFolder
        keyFolder = str(config_static.e_directory_key_folder)
        global pathFileKey
        pathFileKey = directory_main + str(config_static.e_file_key_blocking)
        os.system("touch "+pathFileKey)

def norm(txt):
    return normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII').upper()

def is_vowel(char):
    if char in "AEIOU" : return 1
    else : return 0

def metaPTBR(STRING):
    META_KEY = ""
    CURRENT_POS = 0
    STRING_LENGTH = len(STRING)
    END_OF_STRING_POS = STRING_LENGTH-1
    ORIGINAL_STRING = " " + STRING + "00"
    ORIGINAL_STRING = ORIGINAL_STRING.replace("LH","1")
    ORIGINAL_STRING = ORIGINAL_STRING.replace("NH","3")
    ORIGINAL_STRING = ORIGINAL_STRING.replace("RR","2")
    ORIGINAL_STRING = ORIGINAL_STRING.replace("XC","SS")
    ORIGINAL_STRING = ORIGINAL_STRING.replace("SCH","X")
    ORIGINAL_STRING = ORIGINAL_STRING.replace("TH","T")
    ORIGINAL_STRING = ORIGINAL_STRING.replace("PH","F")
    while (1):
        CURRENT_CHAR = ORIGINAL_STRING[CURRENT_POS]
        if CURRENT_CHAR == "0" : break
        if is_vowel(CURRENT_CHAR) and (CURRENT_POS == 0 or ORIGINAL_STRING[CURRENT_POS - 1] == " "):
            META_KEY += CURRENT_CHAR
            CURRENT_POS += 1
        elif CURRENT_CHAR in "123BDFJKLMPTV" :
            META_KEY += CURRENT_CHAR
            if ORIGINAL_STRING[CURRENT_POS + 1] == CURRENT_CHAR :
                    CURRENT_POS += 2
            else : CURRENT_POS += 1
        else:
            if CURRENT_CHAR == "G":
                    if ORIGINAL_STRING[CURRENT_POS+1] == "E" or ORIGINAL_STRING[CURRENT_POS+1] == "I":
                        META_KEY   += "J"
                        CURRENT_POS += 2
                    elif CURRENT_CHAR == "U":
                        META_KEY   += "G"
                        CURRENT_POS += 2
                    elif CURRENT_CHAR == "R":
                        META_KEY +="GR"
                        CURRENT_POS += 2
                    else:
                        META_KEY   += "G"
                        CURRENT_POS += 2
            elif CURRENT_CHAR == "U":
                if is_vowel(ORIGINAL_STRING[CURRENT_POS-1]) :
                    CURRENT_POS+=1
                    META_KEY+="L"
                else : CURRENT_POS += 1
            elif CURRENT_CHAR == "R":
                if CURRENT_POS==0 or ORIGINAL_STRING[CURRENT_POS - 1] == " " :
                    CURRENT_POS+=1
                    META_KEY+="2"
                elif CURRENT_POS==END_OF_STRING_POS or ORIGINAL_STRING[CURRENT_POS + 1] == " ":
                    CURRENT_POS+=1
                    META_KEY+="2"
                elif is_vowel(ORIGINAL_STRING[CURRENT_POS-1]) and is_vowel(ORIGINAL_STRING[CURRENT_POS+1]) :
                    CURRENT_POS+=1
                    META_KEY+="R"
                else:
                    CURRENT_POS += 1
                    META_KEY+="R"
            elif CURRENT_CHAR == "Z":
                if CURRENT_POS >= len(ORIGINAL_STRING)-1 :
                    CURRENT_POS+=1
                    META_KEY+="S"
                elif ORIGINAL_STRING[CURRENT_POS+1]=="Z" :
                    META_KEY+="Z"
                    CURRENT_POS += 2
                else:
                    CURRENT_POS += 1
                    META_KEY   += "Z"
            elif CURRENT_CHAR == "N":
                if CURRENT_POS >= len(ORIGINAL_STRING)-1 :
                    META_KEY   += "M"
                    CURRENT_POS += 1
                elif ORIGINAL_STRING[CURRENT_POS+1] =="N" :
                    META_KEY   += "N"
                    CURRENT_POS += 2
                else:
                    META_KEY   += "N"
                    CURRENT_POS += 1
            elif CURRENT_CHAR == "S":
                if ORIGINAL_STRING[CURRENT_POS+1]=="S" or CURRENT_POS==END_OF_STRING_POS or ORIGINAL_STRING[CURRENT_POS+1] ==" " :
                    META_KEY += "S"
                    CURRENT_POS += 2
                elif CURRENT_POS==0 or ORIGINAL_STRING[CURRENT_POS-1] == " " :
                    META_KEY += "S"
                    CURRENT_POS += 1
                elif is_vowel(ORIGINAL_STRING[CURRENT_POS-1]) and is_vowel(ORIGINAL_STRING[CURRENT_POS+1]) :
                    META_KEY += "Z"
                    CURRENT_POS += 1
                elif ORIGINAL_STRING[CURRENT_POS+1] =="C" and (ORIGINAL_STRING[CURRENT_POS+2]=="E" or ORIGINAL_STRING[CURRENT_POS+2]=="I") :
                    META_KEY += "S"
                    CURRENT_POS += 3
                elif ORIGINAL_STRING[CURRENT_POS+1] =="C" and (ORIGINAL_STRING[CURRENT_POS+2]=="A" or ORIGINAL_STRING[CURRENT_POS+2]=="O" or ORIGINAL_STRING[CURRENT_POS+2]=="U") :
                    META_KEY += "SC"
                    CURRENT_POS += 3
                else:
                    META_KEY   += "S"
                    CURRENT_POS += 1
            elif CURRENT_CHAR == "X":
                if ORIGINAL_STRING[CURRENT_POS-1] =="E" and CURRENT_POS==1 :
                    META_KEY += "Z"
                    CURRENT_POS += 1
                elif ORIGINAL_STRING[CURRENT_POS-1] =="I" and CURRENT_POS==1 :
                    META_KEY += "X"
                    CURRENT_POS += 1
                elif is_vowel(ORIGINAL_STRING[CURRENT_POS-1]) and CURRENT_POS==1 :
                    META_KEY += "KS"
                    CURRENT_POS += 1
                else:
                    META_KEY += "X"
                    CURRENT_POS += 1
            elif CURRENT_CHAR == "C":
                if ORIGINAL_STRING[CURRENT_POS + 1] == "E" or ORIGINAL_STRING[CURRENT_POS + 1] == "I":
                    META_KEY   += "S"
                    CURRENT_POS += 2
                elif ORIGINAL_STRING[CURRENT_POS + 1]=="H" :
                    META_KEY   += "X"
                    CURRENT_POS += 2
                else:
                    META_KEY   += "K"
                    CURRENT_POS += 1
            elif CURRENT_CHAR == "H":
                if is_vowel(ORIGINAL_STRING[CURRENT_POS + 1]) :
                    META_KEY += ORIGINAL_STRING[CURRENT_POS + 1]
                    CURRENT_POS += 2
                else:
                    CURRENT_POS += 1
            elif CURRENT_CHAR == "Q":
                if ORIGINAL_STRING[CURRENT_POS + 1] == "U" :
                  CURRENT_POS += 2
                else :
                    CURRENT_POS += 1
                META_KEY += "K"
            elif CURRENT_CHAR == "W":
                if is_vowel(ORIGINAL_STRING[CURRENT_POS + 1]) :
                    META_KEY   += "V"
                    CURRENT_POS += 2
                else:
                    META_KEY   += "U"
                    CURRENT_POS += 2
            else :
                CURRENT_POS += 1
    return META_KEY

def bloom(registro, tamanho):

    v_registro = []

    registro = str(' ') + registro
    registro = registro + str(' ')

    for i in range(tamanho):
        v_registro[i:i+1] = "0"

    for i in range(len(registro)-1):
        a = hashlib.md5(registro[i:i+2]).hexdigest()
        b = hashlib.sha1(registro[i:i+2]).hexdigest()

        a1 = int(a,16)%tamanho
        v_registro[a1:a1 +1]="1"
        a2 = (int(a,16) + int(b,16))%tamanho
        v_registro[a2:a2 +1]="1"
        bits_registro = ''.join(v_registro)
    return bits_registro

def writeFiles(line):
    if(status_blocking):
        try:
            for files in line[1:][0]:
                if files != "":
                    l = open(outputFolder+"tmp_/"+ files, 'a')
                    l.write(str(line[0]) + "\n")
                    l.close()
        except Exception:
            print "ERROR WRITING THIS LINE: " + str(line)
    else:
        l = open(outputFolder+"SINGLE_BLOCK.bloom", 'a')
        l.write(str(line) + "\n")

def blockingAndBloom(line_received):

    line_received = line_received.split(";")

    estado = str(line_received[col_st])
    try:
        cidade = str(line_received[col_mr])
    except Exception:
        cidade=""

    nome = str(line_received[col_n])
    dataNascimento = str(line_received[col_bd])
    index = str(line_received[col_i])
    sexo = str(line_received[col_g])
    nomeMae = str(line_received[col_mn])

    data = dataNascimento
    missingFlag = 0

    if(status_name):
        if nome == "NA":
            missingFlag = 1
            array_of_bits = ";"
        else:
            array_of_bits = bloom(nome, size_bloom_col_n) + ";"
    else:
        array_of_bits += ";"

    if(status_birth_date):
        if data == "NA":
            missingFlag = 1
            array_of_bits += ";"
        else:
            array_of_bits += bloom(data, size_bloom_col_bd) + ";"
    else:
        array_of_bits += ";"

    if(status_mother_name):
        if nomeMae == "NA":
            missingFlag = 1
            array_of_bits += ";"
        else:
            array_of_bits += bloom(nomeMae, size_bloom_col_mn) + ";"
    else:
        array_of_bits += ";"

    if(status_municipality_residence):
        if cidade == "NA":
            missingFlag = 1
            array_of_bits += ""
        else:
            array_of_bits += bloom(cidade, size_bloom_col_mr)
    else:
        array_of_bits += ""

    if(status_gender):
        if sexo == "NA":
            missingFlag = 1
            array_of_bits += ""
        else:
            array_of_bits += bloom(sexo, size_bloom_col_mr)
    else:
        array_of_bits += ""

    c=collections.Counter(array_of_bits)
    uns=c.get('1')
    p = config.c_cutoff_result
    if not missingFlag:
        bmin = p*uns/(10000*2 -p)
        bmax = ((10000*2*uns - p*uns)/p) + 1

    else:
        bmin = 0
        bmax = 999

    extracao2 = str(index) + ";" + str(uns) + ";" + str(bmin) + ";" + str(bmax)  + ";" + str(array_of_bits)

    if(status_blocking):

        diaMes = str(dataNascimento[6:8]) + str(dataNascimento[4:6])
        diaAno = str(dataNascimento[6:8]) + str(dataNascimento[:4])
        mesAno = str(dataNascimento[4:6]) + str(dataNascimento[:4])

        # Slicing name
        nomeSeparado = nome.split(" ")
        primeiroNome = nomeSeparado[0]
        ultimoNome = nomeSeparado[(len(nomeSeparado)-1)]

        nomeSeparadoMae = nomeMae.split(" ")
        primeiroNomeMae = nomeSeparadoMae[0]
        ultimoNomeMae = nomeSeparadoMae[(len(nomeSeparadoMae)-1)]

        # Slicing date
        dataSeparada = dataNascimento.split("-")
        try:
            anoNascimento = dataSeparada[0]
        except Exception:
            anoNascimento = "9999"

        if(status_larger_base):
            if(config.blocking_type_1):
                primeiroNome = metaPTBR(primeiroNome)
                ultimoNome = metaPTBR(ultimoNome)
                parte1 = primeiroNome + diaMes
                parte2 = primeiroNome + diaAno
                parte3 = primeiroNome + mesAno
                parte4 = ultimoNome + diaMes
                parte5 = ultimoNome + diaAno
                parte6 = ultimoNome + mesAno
                partes = [parte1, parte2, parte3, parte4, parte5, parte6]
            elif(config.blocking_type_2):
                primeiroNome = metaPTBR(primeiroNome)
                ultimoNome = metaPTBR(ultimoNome)
                primeiroNomeMae = metaPTBR(primeiroNomeMae)
                ultimoNomeMae = metaPTBR(ultimoNomeMae)
                parte1 = primeiroNome + primeiroNomeMae + diaMes
                parte2 = primeiroNome + primeiroNomeMae + diaAno
                parte3 = primeiroNome + primeiroNomeMae + mesAno
                parte4 = ultimoNome + ultimoNomeMae + diaMes
                parte5 = ultimoNome + ultimoNomeMae + diaAno
                parte6 = ultimoNome + ultimoNomeMae + mesAno
                partes = [parte1, parte2, parte3, parte4, parte5, parte6]

            try:
                estrutura = (extracao2, partes)
                return estrutura

            except Exception:
                print "Erro" + partes

        elif(status_smaller_base):
            if(config.blocking_type_1):
                primeiroNome = metaPTBR(primeiroNome)
                ultimoNome = metaPTBR(ultimoNome)
                parte1 = primeiroNome + diaMes
                parte2 = primeiroNome + diaAno
                parte3 = primeiroNome + mesAno
                parte4 = ultimoNome + diaMes
                parte5 = ultimoNome + diaAno
                parte6 = ultimoNome + mesAno
                partes = [parte1, parte2, parte3, parte4, parte5, parte6]
            elif(config.blocking_type_2):
                primeiroNome = metaPTBR(primeiroNome)
                ultimoNome = metaPTBR(ultimoNome)
                primeiroNomeMae = metaPTBR(primeiroNomeMae)
                ultimoNomeMae = metaPTBR(ultimoNomeMae)
                parte1 = primeiroNome + primeiroNomeMae + diaMes
                parte2 = primeiroNome + primeiroNomeMae + diaAno
                parte3 = primeiroNome + primeiroNomeMae + mesAno
                parte4 = ultimoNome + ultimoNomeMae + diaMes
                parte5 = ultimoNome + ultimoNomeMae + diaAno
                parte6 = ultimoNome + ultimoNomeMae + mesAno
                partes = [parte1, parte2, parte3, parte4, parte5, parte6]

            try:
                estrutura = (extracao2, partes)
                return estrutura

            except Exception:
                print "Erro" + partes
    else:
        return extracao2

# Initializing functions

set_variables()
create_path()
flagl = 1
flags = 1

while(flagl or flags):

    if(status_larger_base and flagl):
        print "MAIN PROGRAM MESSAGE (encoding):             Starting round over larger base  (1)"
        set_variables_larger()
        flagl = 0
	rounds = 0

        if(status_smaller_base == 0):
            flags = 0
    elif(status_smaller_base and flags):
        print "MAIN PROGRAM MESSAGE (encoding):             Starting round over smaller base  (2)"
        set_variables_smaller()
        flags = 0
        flagl = 0
	rounds = 1
    entradaRDD = sc.textFile(input_file, partitioning).cache()
    arquivoFinal = entradaRDD.map(blockingAndBloom).map(writeFiles).collect()

fim = time.time()
approx_time = fim - ini
print "MAIN PROGRAM MESSAGE (encoding):             Encoding completed in: " + str(approx_time)
