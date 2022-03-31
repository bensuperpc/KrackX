# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles/appkrackx_autogen.dir/AutogenUsed.txt"
  "CMakeFiles/appkrackx_autogen.dir/ParseCache.txt"
  "appkrackx_autogen"
  )
endif()
