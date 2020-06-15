file(REMOVE_RECURSE
  "../lib/libpacmensl.pdb"
  "../lib/libpacmensl.dylib"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/PACMENSL.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
