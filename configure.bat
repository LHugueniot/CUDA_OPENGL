@echo off
setlocal EnableDelayedExpansion

set arg1=%1

if [%1]==[] goto USAGE

if %arg1% == release (goto RELEASE)
if %arg1% == debug (goto DEBUG)
goto USAGE

:RELEASE
@echo "Configuring release build"
mkdir b_release
cd b_release
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=%VCPKG%/scripts/buildsystems/vcpkg.cmake
cd ..
goto END

:DEBUG
@echo "Configuring debug build"
mkdir b_debug
cd b_debug
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=%VCPKG%/scripts/buildsystems/vcpkg.cmake
cd ..
goto END

:USAGE
@echo "Specify build type: 'release'|'debug'" 

:END