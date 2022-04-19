cd %1
mkdir build
cd build
rem ::cmake .. -G Ninja -DCMAKE_CUDA_FLAGS="-arch=sm_30" 
cmake .. -G Ninja
cd ..