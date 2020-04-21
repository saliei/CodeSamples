CC := gcc
CFLAGS := -Wall -Wextra -g -O3

SRC := $(wildcard *.c)
OBJ := $(SRC:.c=.o)
EXE := test.x

all: $(EXE)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.x:
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: all clean run mem

clean:
	rm -f $(OBJ) $(EXE)

run: $(EXE)
	./$(EXE)

mem: $(EXE)
	valgrind --leak-check=full -s ./$(EXE) 

test.x: hash_table.o test.o
