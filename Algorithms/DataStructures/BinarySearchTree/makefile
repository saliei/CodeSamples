CC := g++
CFLAGS := -std=c++17 -g -Wall -Wextra -O3
INC := .

PRJ := test 
SRC := $(wildcard *.cpp) 
OBJ := $(SRC:%.cpp=%.o)
EXE := $(PRJ:%=%.x)

all: $(EXE)

%.x: %.o
	$(CC) $(CFLAGS) $^ -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INC)

run: all
	./$(EXE)

mem: all
	valgrind --leak-check=full -s ./$(EXE) 

clean:
	rm -f *.o *.x 

.PHONY: all run clean mem
