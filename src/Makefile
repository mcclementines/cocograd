CC = clang
CXX = clang
INCLUDES = #-I
CFLAGS = -g -Wall $(INCLUDES)
CXXFLAGS = -g -Wall $(INCLUDES)
LDFLAGS = -g
LDLIBS = #-L

libcocograd.a: grad.o neuralnet.o
	ar rcs libcocograd.a grad.o neuralnet.o

neuralnet.o: neuralnet.c neuralnet.h

grad.o: grad.c grad.h 

.PHONY: all
all: clean libcocograd.a

.PHONY: clean
clean:
	rm -f *.o *.a
