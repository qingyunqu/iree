// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/dylib/registration/driver_module_sync.h"

#include <inttypes.h>

#include "iree/hal/local/loaders/legacy_library_loader.h"
#include "iree/hal/local/sync_driver.h"

// TODO(#4298): remove this driver registration and wrapper.
// By having a single iree/hal/local/registration that then has the loaders
// added to it based on compilation settings we can have a single set of flags
// for everything.

#define IREE_HAL_DYLIB_SYNC_DRIVER_ID 0x53444C4Cu  // SDLL

static iree_status_t iree_hal_dylib_sync_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  static const iree_hal_driver_info_t default_driver_info = {
      .driver_id = IREE_HAL_DYLIB_SYNC_DRIVER_ID,
      .driver_name = iree_string_view_literal("dylib-sync"),
      .full_name = iree_string_view_literal("AOT compiled dynamic libraries"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_dylib_sync_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  if (driver_id != IREE_HAL_DYLIB_SYNC_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }

  iree_hal_sync_device_params_t default_params;
  iree_hal_sync_device_params_initialize(&default_params);

  iree_hal_executable_loader_t* dylib_loader = NULL;
  iree_status_t status =
      iree_hal_legacy_library_loader_create(allocator, &dylib_loader);
  iree_hal_executable_loader_t* loaders[1] = {dylib_loader};

  if (iree_status_is_ok(status)) {
    status = iree_hal_sync_driver_create(
        iree_make_cstring_view("dylib"), &default_params,
        IREE_ARRAYSIZE(loaders), loaders, allocator, out_driver);
  }

  iree_hal_executable_loader_release(dylib_loader);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_dylib_sync_driver_module_register(
    iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_dylib_sync_driver_factory_enumerate,
      .try_create = iree_hal_dylib_sync_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}