// TODO(antiagainst): Re-enable SPIR-V linking once the tensorflow integration
// crash is fixed.
// RUN-disabled: iree-opt -split-input-file -iree-hal-link-target-executables='target=vulkan-spirv'  %s | IreeFileCheck %s
// RUN: iree-opt -split-input-file %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb">

hal.executable private @call_dispatch_0  {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.entry_point @call_dispatch_0 attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_0() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_0
        spv.ExecutionMode @call_dispatch_0 "LocalSize", 32, 1, 1
      }
      hal.interface private @io  {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
      }
    }
  }
}
hal.executable private @call_dispatch_1  {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.entry_point @call_dispatch_1 attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_1() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_1
        spv.ExecutionMode @call_dispatch_1 "LocalSize", 4, 4, 1
      }
      hal.interface private @io  {
        hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
hal.executable private @call_dispatch_2  {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.entry_point @call_dispatch_2 attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_2() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_2
        spv.ExecutionMode @call_dispatch_2 "LocalSize", 32, 1, 1
      }
      hal.interface private @io  {
        hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
      }
    }
  }
}
hal.executable private @call_dispatch_3  {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.entry_point @call_dispatch_3 attributes {interface = @io, ordinal = 0 : index} {
    ^bb0(%arg0: index, %arg1: index, %arg2: index):  // no predecessors
      %c1 = arith.constant 1 : index
      %c56 = arith.constant 56 : index
      %c56_0 = arith.constant 56 : index
      hal.return %c1, %c56, %c56_0 : index, index, index
    }
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_3() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_3
        spv.ExecutionMode @call_dispatch_3 "LocalSize", 8, 2, 2
      }
      hal.interface private @io  {
        hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
hal.executable private @call_dispatch_4  {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
    hal.executable.entry_point @call_dispatch_4 attributes {interface = @io, ordinal = 0 : index}
    builtin.module {
      spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
        spv.func @call_dispatch_4() "None" {
          spv.Return
        }
        spv.EntryPoint "GLCompute" @call_dispatch_4
        spv.ExecutionMode @call_dispatch_4 "LocalSize", 2, 8, 1
      }
      hal.interface private @io  {
        hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// Two groups should be created, according to their interfaces.

//      CHECK: hal.executable private @linking_linked_vulkan_0 {
// CHECK-NEXT:   hal.interface public @io_0 {
// CHECK-NEXT:     hal.interface.binding public @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
// CHECK-NEXT:     hal.interface.binding public @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
// CHECK-NEXT:     hal.interface.binding public @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// CHECK-NEXT:   }
// CHECK-NEXT:   hal.executable.variant public @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
// CHECK-NEXT:     hal.executable.entry_point public @call_dispatch_1 attributes {interface = @io_0, ordinal = 0 : index}
// CHECK-NEXT:     hal.executable.entry_point public @call_dispatch_3 attributes {interface = @io_0, ordinal = 1 : index}
// CHECK-NEXT:     hal.executable.entry_point public @call_dispatch_4 attributes {interface = @io_0, ordinal = 2 : index}
// CHECK-NEXT:     module  {
// CHECK-NEXT:       spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
// CHECK-NEXT:         spv.func @call_dispatch_1() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_1
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_1 "LocalSize", 4, 4, 1
// CHECK-NEXT:         spv.func @call_dispatch_3() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_3
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_3 "LocalSize", 8, 2, 2
// CHECK-NEXT:         spv.func @call_dispatch_4() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_4
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_4 "LocalSize", 2, 8, 1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

//      CHECK: hal.executable private @linking_linked_vulkan {
// CHECK-NEXT:   hal.interface public @io_0 {
// CHECK-NEXT:     hal.interface.binding public @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
// CHECK-NEXT:     hal.interface.binding public @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
// CHECK-NEXT:   }
// CHECK-NEXT:   hal.executable.variant public @vulkan_spirv_fb, target = #executable_target_vulkan_spirv_fb {
// CHECK-NEXT:     hal.executable.entry_point public @call_dispatch_0 attributes {interface = @io_0, ordinal = 0 : index}
// CHECK-NEXT:     hal.executable.entry_point public @call_dispatch_2 attributes {interface = @io_0, ordinal = 1 : index}
// CHECK-NEXT:     module  {
// CHECK-NEXT:       spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
// CHECK-NEXT:         spv.func @call_dispatch_0() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_0
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_0 "LocalSize", 32, 1, 1
// CHECK-NEXT:         spv.func @call_dispatch_2() "None" {
// CHECK-NEXT:           spv.Return
// CHECK-NEXT:         }
// CHECK-NEXT:         spv.EntryPoint "GLCompute" @call_dispatch_2
// CHECK-NEXT:         spv.ExecutionMode @call_dispatch_2 "LocalSize", 32, 1, 1
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
