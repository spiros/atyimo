/* 
## FEDERAL UNIVERSITY OF BAHIA (UFBA)
## ATYIMOLAB (www.atyimolab.ufba.br)
## University College London
## Denaxas Lab (www.denaxaslab.org)

# File:           $cb.c$
# Version:        $v1$
# Last changed:   $Date: 2017/12/04 12:00:00$
# Purpose:        $C function to calculate the Dice coefficient$
# Author:         Robespierre Pita and Clicia Pinto and Marcos Barreto and Spiros Denaxas

# Usage:  gcc --shared cb.c -o libcb.so

# Comments:

*/

int calculate_h(char *bloom1, char *bloom2, int size) {
	int h =0;
	for (int i = 0; i < size; i++) {
		if (bloom1[i] == '1') {
			if (bloom2[i] == '1') {
				h++;
			}
		}
	}
	return h;
}
