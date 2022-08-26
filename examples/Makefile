CC = clang
CXX = clang
INCLUDES = -I../src
CFLAGS = -g -Wall $(INCLUDES)
CXXFLAGS = -g -Wall $(INCLUDES)
LDFLAGS = -g
LDLIBS = -L../src -lcocograd

.PHONY: all
all: clean value_operations simple_neural_net

simple_neural_net: simple_neural_net.o

simple_neural_net.o: simple_neural_net.c

value_operations: value_operations.o

value_operations.o: value_operations.c

.PHONY: run
run: $(prog)
	./$(prog)

# .PHONY: valgrind
# valgrind: value_operations
# 	valgrind --leak-check=yes ./value_operations

.PHONY: clean
clean:
	rm -f *.o *.a value_operations simple_neural_net
