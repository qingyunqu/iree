// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-llvmgpu-remove-single-iteration-loop))))' %s | IreeFileCheck %s

// CHECK-LABEL: func @dispatch_0()
hal.executable private @dispatch_0  {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @dispatch_0 attributes {
      interface = @io,
      ordinal = 0 : index,
      workgroup_size = [64: index, 1: index, 1:index]}
    builtin.module {
      builtin.func @dispatch_0() {
        %c2 = arith.constant 2 : index
        %c256 = arith.constant 256 : index
        //     CHECK: %[[C250:.+]] = arith.constant 250 : index
        %c250 = arith.constant 250 : index
        %tidx = "gpu.thread_id"() {dimension = "x"} : () -> index
        %tidy = "gpu.thread_id"() {dimension = "y"} : () -> index
        // CHECK-NOT: scf.for
        //     CHECK: gpu.barrier
        scf.for %arg3 = %tidy to %c2 step %c2 {
          %0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%tidx]
          scf.for %arg4 = %0 to %c256 step %c256 {
             gpu.barrier
          }
        }
        // The inner loop doesn't always execute once so it cannot be removed.
        //     CHECK: scf.for %{{.*}} = %{{.*}} to %[[C250]] step %[[C250]]
        //     CHECK:   gpu.barrier
        scf.for %arg3 = %tidy to %c2 step %c2 {
          %0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%tidx]
          scf.for %arg4 = %0 to %c250 step %c250 {
             gpu.barrier
          }
        }
        return
      }
    }
  }
}
