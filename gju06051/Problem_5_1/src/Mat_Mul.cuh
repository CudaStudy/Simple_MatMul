#include "/usr/local/cuda-10.2/targets/x86_64-linux/include/cuda_runtime.h"
#include "/usr/local/cuda-10.2/targets/x86_64-linux/include/device_launch_parameters.h"
#include "/usr/local/cuda-10.2/targets/x86_64-linux/include/crt/device_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INDEX2ROW(_index, _width) (int)((_index) / (_width))
#define INDEX2COL(_index, _width) (int)((_index) % (_width))
#define ID2INDEX(_row, _col, _width) (((_row) * (_width)) + (_col))

#define BLOCK_SIZE 16