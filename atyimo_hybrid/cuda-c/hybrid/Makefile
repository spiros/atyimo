MKL_PATH_INCLUDE=$(MKLROOT)/include
MKL_PATH_LIB=$(MKLROOT)/lib/em64t

MKL_LIBS_64=-lmkl_lapack -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5
MKL_LIBS_64=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

COMMONDIR  := ../common

LIB_DIR     := ../lib

LIB_PATHS=-L$(MKL_PATH_LIB) -L$(LIB_DIR)

NVCC=nvcc -ccbin=gcc-4.8
CC=gcc

sources.cu=linkage.cu
sources.c=

CFLAGS = --compiler-options -fno-strict-aliasing -D_FORCE_INLINES -DUNIX -DADD_ -arch sm_20 -O3 -lm -lstdc++

libs = $(LIB_PATHS) -lcublas -lcudart $(MKL_LIBS_64) -lpthread

exec = linkage

objects = $(sources.c:.c=.o) $(sources.cu:.cu=.o)


all: $(exec)

$(exec): $(includes) $(objects)
	$(NVCC) $(objects) $(CFLAGS) -o $@ $(libs)

main.o:linkage.cu
	$(NVCC) $(CFLAGS) -c $^

%.o:%.cu
	$(NVCC) -Xcompiler -fopenmp -lgomp $(CFLAGS) -c $^

%.o:%.c
	$(CC) $(CFLAGS) -c $^

clean:
	rm -f linkage *.txt $(objects) *~
