if(NOT DEFINED ROPTLIB_SOURCE_DIR)
  message(FATAL_ERROR "ROPTLIB_SOURCE_DIR is required")
endif()

set(my_matrix "${ROPTLIB_SOURCE_DIR}/Others/MyMatrix.cpp")
if(NOT EXISTS "${my_matrix}")
  message(FATAL_ERROR "Missing ROPTLIB source file: ${my_matrix}")
endif()

file(READ "${my_matrix}" content)
string(REPLACE
  "if (eigenvalues <= 0)"
  "if (eigenvalues[i] <= 0)"
  patched
  "${content}")

if(NOT patched STREQUAL content)
  file(WRITE "${my_matrix}" "${patched}")
endif()
