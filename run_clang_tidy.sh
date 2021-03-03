cd cmake_build
run-clang-tidy -header-filter='.*' \
    -checks='-*,clang-diagnostic-*,clang-analyzer-*,bugprone*,modernize*,performance*,
    -modernize-pass-by-value,-modernize-use-auto,-modernize-use-using,-modernize-use-trailing-return-type,-bugprone-exception-escape'

