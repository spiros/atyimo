#!/bin/bash

## FEDERAL UNIVERSITY OF BAHIA (UFBA)
## ATYIMOLAB (www.atyimolab.ufba.br)
## University College London
## Denaxas Lab (www.denaxaslab.org)

if type java; then
    echo "Found Java executable in PATH"
    _java=java
elif [ -n "$JAVA_HOME" ] && [ -x "$JAVA_HOME/bin/java" ];  then
    echo "Found java executable in JAVA_HOME"     
    _java="$JAVA_HOME/bin/java"
else
    echo "Java not found"
    exit 0
fi
echo "Where do you want to install Spark?  "
read sparkpath
if [ -e $sparkpath ] && [ -w $sparkpath ] && [ -d $sparkpath ]; then
	mkdir -p $sparkpath/.tmp
	wget http://ftp.unicamp.br/pub/apache/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz --directory-prefix=$sparkpath/.tmp/
	tar -xf $sparkpath/.tmp/spark-2.2.0-bin-hadoop2.7.tgz -C $sparkpath
echo "Where is your AtyImo git clone?  "
read atyimopath
else
	echo "Invalid path!"
fi
if [ -e $atyimopath ] && [ -w $atyimopath ] && [ -d $atyimopath ]; then echo "Creating AtyImo script at $atyimopath/SCRIPT.sh";	echo "$sparkpath/bin/spark-submit $atyimopath/01_preprocessing.py" >> $sparkpath/SCRIPT.sh; echo "$sparkpath/bin/spark-submit $atyimopath/02_createBlockKey.py" >> $sparkpath/SCRIPT.sh; echo "$sparkpath/bin/spark-submit $atyimopath/03_writeBlocks.py" >> $sparkpath/SCRIPT.sh; echo "$sparkpath/bin/spark-submit $atyimopath/04_encoding_blocking.py" >> $sparkpath/SCRIPT.sh; echo "$sparkpath/bin/spark-submit $atyimopath/05_correlation.py" >> $sparkpath/SCRIPT.sh; echo "$sparkpath/bin/spark-submit $atyimopath/06_dedupByKey.py" >> $sparkpath/SCRIPT.sh; echo "$sparkpath/bin/spark-submit $atyimopath/06_geraDataMart.py" >> $sparkpath/SCRIPT.sh; echo "compiling shared lib to compare Bloom filters"; gcc $atyimopath/cb.c --shared -o $atyimopath/libcb.so;echo "you're ready to go"; else echo "this is a invalid path"; fi
