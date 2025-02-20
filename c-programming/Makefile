CC = mpicc
CFLAGS = -O2 -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib -lopenblas -llapacke -llapack -lm

TARGETS = tsqr_mpi scaling

all: $(TARGETS)

# Build tsqr_mpi: depends only on tsqr_mpi.c
tsqr_mpi: tsqr_mpi.o
	$(CC) $(CFLAGS) -o tsqr_mpi tsqr_mpi.o $(LDFLAGS)

# Build scaling: depends only on scaling.c
scaling: scaling.o
	$(CC) $(CFLAGS) -o scaling scaling.o $(LDFLAGS)

# Generic rule: %.o depends on the corresponding %.c file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS) scaling_results.txt
