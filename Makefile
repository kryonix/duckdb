.PHONY: all opt unit clean debug release test unittest allunit docs doxygen format sqlite imdb

all: release
opt: release
unit: unittest
imdb: third_party/imdb/data

GENERATOR=
FORCE_COLOR=
WARNINGS_AS_ERRORS=
ifeq ($(GEN),ninja)
	GENERATOR=-G "Ninja"
	FORCE_COLOR=-DFORCE_COLORED_OUTPUT=1
endif
ifeq (${TREAT_WARNINGS_AS_ERRORS}, 1)
	WARNINGS_AS_ERRORS=-DTREAT_WARNINGS_AS_ERRORS=1
endif

clean:
	rm -rf build

debug:
	mkdir -p build/debug && \
	cd build/debug && \
	cmake $(GENERATOR) $(FORCE_COLOR) ${WARNINGS_AS_ERRORS} -DCMAKE_BUILD_TYPE=Debug ../.. && \
	cmake --build .

release:
	mkdir -p build/release && \
	cd build/release && \
	cmake $(GENERATOR) $(FORCE_COLOR) ${WARNINGS_AS_ERRORS} -DCMAKE_BUILD_TYPE=Release ../.. && \
	cmake --build .

unittest: debug
	build/debug/test/unittest
	build/debug/tools/sqlite3_api_wrapper/test_sqlite3_api_wrapper

allunit: release # uses release build because otherwise allunit takes forever
	build/release/test/unittest "*"

docs:
	mkdir -p build/docs && \
	doxygen Doxyfile

doxygen: docs
	open build/docs/html/index.html

amalgamation:
	mkdir -p build/amalgamation && \
	python scripts/amalgamation.py && \
	cd build/amalgamation && \
	cmake $(GENERATOR) $(FORCE_COLOR) -DAMALGAMATION_BUILD=1 -DCMAKE_BUILD_TYPE=Release ../.. && \
	cmake --build .

test_compile: # test compilation of individual cpp files
	python scripts/amalgamation.py --compile

format:
	python scripts/format.py

third_party/sqllogictest:
	git clone --depth=1 https://github.com/cwida/sqllogictest.git third_party/sqllogictest

third_party/imdb/data:
	wget -i "http://download.duckdb.org/imdb/list.txt" -P third_party/imdb/data

sqlite: release | third_party/sqllogictest
	git --git-dir third_party/sqllogictest/.git pull
	./build/release/test/unittest "[sqlitelogic]"

sqlsmith: debug
	./build/debug/third_party/sqlsmith/sqlsmith --duckdb=:memory:
