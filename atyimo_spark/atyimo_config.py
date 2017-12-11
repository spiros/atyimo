#!/bin/env python
# coding: utf-8

## FEDERAL UNIVERSITY OF BAHIA (UFBA)
## ATYIMOLAB (www.atyimolab.ufba.br)
## University College London
## Denaxas Lab (www.denaxaslab.org)

# File:           $atyimo_config.py$
# Version:        $v1$
# Last changed:   $Date: 2017/12/04 12:00:00 $
# Purpose:        Data sets dependent configuration for AtyImo's modules
# Author:         Robespierre Pita and Clicia Pinto and Marcos Barreto and Spiros Denaxas

# Usage:  /path/to/python/atyimo_config.py

# Comments:
# You must set default_folder to your preferred location (execution folder).
# You must also set pp_larger_input_file and pp_smaller_input_file to the corresponding data sets to be linked.
# You must check date format and gender type for each input data set and perform the necessary adjustments.
# You must match items' position according to the column names in both data sets

status_larger_base = 1  # Boolean indicating execution over a large data set
status_smaller_base = 1 # Boolean indicating execution over a small data set
default_folder = "/PATH/TO/WORK/DIRECTORY/" # Main execution folder

# Set everything to be used in preprocessing module

# Larger base - larger input file to the preprocessing module
pp_larger_input_file = "/PATH/TO/larger_database.csv"

# Boolean indicating the existence of labels (column names) in the larger data set
pp_larger_status_label = 1
# Boolean indicating the existence of these variables in the larger data set
pp_larger_status_code = 1                               # Code
pp_larger_status_name = 1								# Name
pp_larger_status_birth_date = 1                         # Date of birth
pp_larger_status_gender = 1                             # Gender
pp_larger_status_mother_name = 1                        # Mother's name
pp_larger_status_municipality_residence = 1     		# Municipality of residence
pp_larger_status_state = 0                              # State (federation unit) - not used

# Boolean indicating the date format in the larger data set
pp_larger_type_date1 = 0 # yearmonthday examples: 19990414  <- STANDARD
pp_larger_type_date2 = 0 # daymonthyear examples: 14041999
pp_larger_type_date3 = 0 # day<"-"||"/">month<"-"||"/">year examples: 14/mar/08 or 14/mar/2008 or 1-Dec-2006 or 1-Dec-06
pp_larger_type_date4 = 0 # year-month-day examples: 1963-01-29  year<"-"||"/">month<"-"||"/">day examples: 1999/mar/08 or 08/mar/29 or 2006-Dec-1 or 06-Dec-01
pp_larger_type_date5 = 1 # month<"-"||"">year examples: 11/13/1965 or nov/13/1965 or 1965-Nov-13

# Boolean indicating the gender type in the larger data set
pp_larger_type_gender1 = 1 # 1 for male and 2 for female  <- STANDARD
pp_larger_type_gender2 = 0 # M for male and F for female
pp_larger_type_gender3 = 0 # 1 for male and 3 for female

# Boolean indicating the municipality type in the larger data set
pp_larger_type_municipality1 = 1  # IBGE code with 4 or 5 digits <- STANDARD
pp_larger_type_municipality2 = 0  # IBGE code with 6 or 7 digits (state included)
pp_larger_type_municipality3 = 0  # Municipality name

# Items position in the larger data set
pp_larger_col_c = 0 	    # Code
pp_larger_col_n = 1         # Name
pp_larger_col_mn = 2		# Mother's name
pp_larger_col_bd = 3        # Date of birth
pp_larger_col_g =  4   		# Gender
pp_larger_col_mr = 5      	# Municipality of residence
pp_larger_col_st = -1  		# State (not used)

# Smaller base - smaller input file to the preprocessing module
pp_smaller_input_file = "/PATH/TO/smaller_database.csv"

# Boolean indicating the existence of labels (column names) in the smaller data set
pp_smaller_status_label = 1

# Boolean indicating the existence of these variables in the smaller data set
pp_smaller_status_code = 1					 # Code
pp_smaller_status_name = 1					 # Name
pp_smaller_status_birth_date = 1			 # Date of birth
pp_smaller_status_gender = 1				 # Gender
pp_smaller_status_mother_name = 1			 # Mother's name
pp_smaller_status_municipality_residence = 1 # Municipality of residence
pp_smaller_status_state = 0					 # State (not used)

# Boolean indicating the date format in the smaller data set
pp_smaller_type_date1 = 0 # yearmonthday examples: 19990414  <- PADRÃO
pp_smaller_type_date2 = 0 # daymonthyear examples: 14041999
pp_smaller_type_date3 = 0 # day<"-"||"/">month<"-"||"/">year examples: 14/mar/08 or 14/mar/2008 or 1-Dec-2006 or 1-Dec-06 (only for years after 1999)
pp_smaller_type_date4 = 0 # year-month-day examples: 1963-01-29  year<"-"||"/">month<"-"||"/">day examples: 1999/mar/08 or 08/mar/29 or 2006-Dec-1 or 06-Dec-01
pp_smaller_type_date5 = 1 # month<"-"||"/">day<"-"||"/">year examples: 11/13/1965 or nov/13/1965 or 1965-Nov-13

# Boolean indicating the gender type in the smaller data set
pp_smaller_type_gender1 = 1  # 1 for male and 2 for female  <- STANDARD
pp_smaller_type_gender2 = 0  # M for male and F for female
pp_smaller_type_gender3 = 0  # 1 for male and 3 for female

#Boolean indicating the municipality type of smaller base
pp_smaller_type_municipality1 = 1  # IBGE code with 4 or 5 digits <- STANDARD
pp_smaller_type_municipality2 = 0  # IBGE code with 6 or 7 digits (state included)
pp_smaller_type_municipality3 = 0  # Municipality name

# Items position in the smaller data set
pp_smaller_col_c = 0		# Code
pp_smaller_col_n = 1		# Name
pp_smaller_col_mn = 2		# Mother name
pp_smaller_col_bd = 3		# Date of birth date
pp_smaller_col_g = 4		# Gender
pp_smaller_col_mr = 5		# Municipality of residence
pp_smaller_col_st = -1		# State (not used)

##----##----##----##----##----##----##----##----##----##----##----##----##----##--------##----##----##----##----##----##----

# Set parameters used in the correlation module
# Main cutoff (Dice)
c_cutoff_result = 8000

##----##----##----##----##----##----##----##----##----##----##----##----##----##--------##----##----##----##----##----##----

# Set parameters used in the encoding module
# Boolean indicating the use of blocking
# type_1 and type_2 mean different predicate blocking strategies
e_status_blocking = 0

# pName - given name, uName - last surname (the same for pMotherName and uMotherName)
# db - day of birth, mb - month of birth, yb - year of birth
blocking_type_1 = 0 	# (pName AND ((db AND mb) OR (mb OR yb) OR (db AND yb))) OR (uName AND (db AND mb) OR (mb AND yb) OR (db AND yb)))
blocking_type_2 = 0		# (pName AND pMotherName AND ((db AND mb) OR (mb OR yb) OR (db AND yb))) OR (uName AND uMotherName AND (db AND mb) OR (mb AND yb) OR (db AND yb)))

# Boolean indicating the existence of these variables in the Bloom vector
e_status_name = 1						# Name
e_status_mother_name = 1				# Mother's name
e_status_birth_date = 1					# Date of birth
e_status_gender = 1						# Gender
e_status_municipality_residence = 1		# Municipality of residence
e_status_state = 0						# State (not used)

# Size (weight), in bits, of these variables in the Bloom vector
e_size_bloom_col_n = 50 	# Name
e_size_bloom_col_mn = 50 	# Mother's name
e_size_bloom_col_bd = 40 	# Date of birth
e_size_bloom_col_mr = 20 	# Municipality of residence
e_size_bloom_col_g = 40		# Gender

##----##----##----##----##----##----##----##----##----##----##----##----##----##--------##----##----##----##----##----##----

# Set parameters used in the datamart build module
# Type of resulting datamart. Be free to choose more than one to be generated as AtyImo's outputs.
dm_type1 = 0			# [default] attributes of larger data set ; matched attributes of smaller data set
dm_type2 = 0			# attributes of larger data set ; flag indicating pairing (true of false)
dm_type3 = 0			# attributes of larger database ; flag indicating pairing (true of false) ; matched attributes of smaller data set
dm_type4 = 1			# All matches from the smaller data set with their highest Dice pairs in the larger data set

# Set parameters used in the second linkage round
# Boolean indicating the second round of verification oover the gray area
sr_status_second_round = 0
# higher cutoff
sr_higher_cutoff = 9500.0
