# Tiny AOT client for the aua warm daemon (no Python import).
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic -std=c11
PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin

.PHONY: all clean install test

all: aua-fast

aua-fast: main.c
	$(CC) $(CFLAGS) -o $@ main.c

install: aua-fast
	install -d "$(BINDIR)"
	install -m 755 aua-fast "$(BINDIR)/aua-fast"

clean:
	rm -f aua-fast

test: aua-fast
	./aua-fast --help >/dev/null
	@echo "aua-fast built OK ($$(wc -c < aua-fast | tr -d ' ') bytes)"
