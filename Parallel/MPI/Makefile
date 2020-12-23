MPICC  := mpicc

CFLAGS   := -O2 -g
CPPFLAGS :=
LDFLAGS  :=

INCS :=
LIBS :=

.c.o:
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

all: mpi_collective_open.x

mpi_collective_open.x: mpi_collective_open.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.o test_mpi_io.dat \
	mpi_collective_open.x

.PHONY: clean
