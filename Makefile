CC = gcc
CFLAGS = -std=c17 -O2 -Wall -Wextra -pedantic -ffast-math
LDFLAGS = -lm

PREFIX = /usr/local

.PHONY: all clean test install

all: test_p48

test_p48: src/test_p48.c src/p48.h
	$(CC) $(CFLAGS) -o test_p48 src/test_p48.c $(LDFLAGS)

test: test_p48
	./test_p48

clean:
	rm -f test_p48 *.o

install: test_p48
	cp src/p48.h $(PREFIX)/include/
	@echo "Installed p48.h to $(PREFIX)/include/"
