CC = gcc
CFLAGS = -O3 -fPIC -std=c99 -Wall -Wextra
LDFLAGS = -shared

# Betriebssystem-spezifische Einstellungen
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBEXT = .so
endif
ifeq ($(UNAME_S),Darwin)
    LIBEXT = .so
    LDFLAGS += -undefined dynamic_lookup
endif
ifdef OS  # Windows (MinGW)
    LIBEXT = .dll
    CC = x86_64-w64-mingw32-gcc
endif

TARGET = baillie_psw_complete$(LIBEXT)
SOURCE = baillie_psw_complete.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET) *.o

test: $(TARGET)
	@echo "Running quick sanity check..."
	@echo "Baillie-PSW on small primes:"
	@echo "  should be True: $$(./$(TARGET) 97)"
	@echo "  should be False: $$(./$(TARGET) 561)"

.PHONY: all clean test
