add_definitions(-DSQLITE_OMIT_LOAD_EXTENSION=1)

SET (SHELL_SOURCES shell.c)
if(NOT WIN32)
	add_definitions(-DHAVE_LINENOISE=1)
	SET (SHELL_SOURCES ${SHELL_SOURCES} linenoise.c)
endif()

include_directories(include)
include_directories(../sqlite3_api_wrapper/include)
add_executable(shell ${SHELL_SOURCES})
target_link_libraries(shell sqlite3_api_wrapper_static)
set_target_properties(shell PROPERTIES OUTPUT_NAME duckdb_cli)
set_target_properties(shell PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
