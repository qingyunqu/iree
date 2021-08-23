# How to Run Tests
### Compile IREE
```
cmake -G Ninja -B ./build/ -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ .
cmake --build ./build/
```
* `cmake --build ./build/ --target iree-opt` 可以只编译`iree-opt`，不跑testcase
### Backend List
* cuda
* vmvx
* vulkan-spirv
* dylib-llvm-aot
### Generate and Run .vmfb with CUDA Backend
```
iree-translate -iree-input-type=mhlo -iree-mlir-to-vm-bytecode-module -iree-hal-target-backends=cuda add.mlir -o add.vmfb
```
* 输入为linalg时设置`-iree-input-type=none`
* `--iree-cuda-dump-ptx`可以dump出生成的ptx
* `--print-ir-after-all`可以打印出中间的转换过程
```
iree-run-module --driver=cuda --module_file=./add.vmfb --entry_function=add --function_input="4xf32=[1 2 3 4]" --function_input="4xf32=[2 2 2 2]"
```
### Generate Linalg Dialect
```
iree-opt -iree-mhlo-input-transformation-pipeline test.mlir -o test.linalg.mlir
```
* `--iree-flow-fusion-of-tensor-ops`主要实现了linalg tensor上的fusion
* `--iree-enable-fusion-with-reduction-ops`可以开启reduce上的fusion
### Linalg Dialect -> Affine Dialect
* `DispatchLinalgOnTensors`