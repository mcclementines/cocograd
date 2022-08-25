CC = clang
CXX = clang
INCLUDES = -I../cocograd
CFLAGS = -g -Wall $(INCLUDES)
CXXFLAGS = -g -Wall $(INCLUDES)
LDFLAGS = -g
LDLIBS = -L../cocograd -lcocograd

main: main.o

main.o: main.c

.PHONY: all
all: clean main

.PHONY: run
run: main
	./main

.PHONY: valgrind
valgrind: main
	valgrind --leak-check=yes ./main

.PHONY: clean
clean:
	rm -f *.o *.a main
