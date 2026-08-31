import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_cpp_operator_call_instantiation_smoke(tmp_path):
    binary = _compile_cpp(tmp_path, "add_smoke", _ADD_SMOKE_SOURCE)
    _run([str(binary)])


def test_cpp_returning_call_smoke(tmp_path):
    binary = _compile_cpp(tmp_path, "add_return_smoke", _ADD_RETURN_SMOKE_SOURCE)
    _run([str(binary)])


def test_tuning_cache_round_trip(tmp_path):
    binary = _compile_cpp(tmp_path, "tuning_cache", _TUNING_CACHE_SOURCE)
    cache_path = tmp_path / "tuning.json"
    environment = os.environ.copy()
    environment.update(
        {
            "INFINI_OPS_TUNING_PATH": str(cache_path),
            "INFINI_OPS_TUNING_WARMUP": "2",
            "INFINI_OPS_TUNING_REPEAT": "3",
        }
    )

    _run([str(binary), "initialize", str(cache_path)], env=environment)

    cache = json.loads(cache_path.read_text())
    assert cache["version"] == 1
    assert cache["entries"][0]["best_implementation"] == 7

    _run([str(binary), "lookup", str(cache_path)])

    cache_path.write_text("{")
    _run([str(binary), "miss", str(cache_path)])


def _compile_cpp(tmp_path, name, source_text):
    install_prefix = _install_prefix()
    include_dirs = [install_prefix / "include"]
    if infinirt_root := os.environ.get("INFINI_RT_ROOT"):
        include_dirs.append(Path(infinirt_root) / "include")
    library_dir = _library_dir(install_prefix)
    source = tmp_path / f"{name}.cc"
    binary = tmp_path / name
    source.write_text(source_text)

    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-Werror",
            *(f"-I{include_dir}" for include_dir in include_dirs),
            str(source),
            f"-L{library_dir}",
            "-linfiniops",
            "-linfinirt",
            f"-Wl,-rpath,{library_dir}",
            "-o",
            str(binary),
        ]
    )

    return binary


def _install_prefix():
    prefix = os.environ.get("INFINI_OPS_INSTALL_PREFIX")

    if prefix:
        return Path(prefix)

    pytest.skip("`INFINI_OPS_INSTALL_PREFIX` is not set.")


def _library_dir(prefix):
    for library_dir in (prefix, prefix / "lib", prefix / "lib64"):
        if all(
            (library_dir / name).exists()
            for name in ("libinfiniops.so", "libinfinirt.so")
        ):
            return library_dir

    pytest.skip(f"InfiniOps and InfiniRT libraries were not found under `{prefix}`.")


def _compiler(env_name, default):
    compiler = os.environ.get(env_name, default)

    if not compiler:
        pytest.skip(f"`{env_name}` is not configured.")

    return compiler


def _run(command, *, env=None):
    try:
        subprocess.run(command, check=True, text=True, capture_output=True, env=env)
    except FileNotFoundError as error:
        pytest.skip(f"`{command[0]}` is not available: {error}")
    except subprocess.CalledProcessError as error:
        output = f"{error.stdout}\n{error.stderr}".strip()
        raise AssertionError(output) from error


_TUNING_CACHE_SOURCE = textwrap.dedent(
    r"""
    #include <tuning.h>

    #include <string>

    int main(int argc, char** argv) {
      if (argc != 3) {
        return 2;
      }

      infini::ops::TuningSignature signature;
      signature.tensors.push_back(
          {{2, 3}, infini::ops::DataType::kFloat32});
      signature.scalars.push_back(1.5);

      auto& manager = infini::ops::TuningManager::Instance();
      const std::string mode{argv[1]};

      if (mode == "initialize") {
        manager.InitializeFromEnvironment();
        if (manager.warmup_count() != 2 || manager.repeat_count() != 3) {
          return 1;
        }
        manager.Record("Add", infini::ops::Device::Type::kCpu, signature, 7);
        return 0;
      }

      manager.LoadTuningCache(argv[2]);
      auto implementation = manager.Lookup(
          "Add", infini::ops::Device::Type::kCpu, signature);
      if (mode == "lookup") {
        return implementation == std::optional<std::size_t>{7} ? 0 : 1;
      }
      if (mode == "miss") {
        return implementation.has_value() ? 1 : 0;
      }
      return 2;
    }
    """
).lstrip()


_ADD_SMOKE_SOURCE = textwrap.dedent(
    r"""
    #include <infini/ops.h>

    #include <cmath>

    int main() {
      float input_data[3] = {1.0f, 2.0f, 3.0f};
      float other_data[3] = {4.0f, 5.0f, 6.0f};
      float output_data[3] = {0.0f, 0.0f, 0.0f};

      const infini::ops::Tensor::Shape shape{3};
      const infini::ops::Device device{infini::ops::Device::Type::kCpu};
      const infini::ops::DataType data_type{infini::ops::DataType::kFloat32};

      infini::ops::Tensor input(input_data, shape, data_type, device);
      infini::ops::Tensor other(other_data, shape, data_type, device);
      infini::ops::Tensor output(output_data, shape, data_type, device);
      infini::ops::Handle handle;
      infini::ops::Config config;

      infini::ops::Add::Call(handle, config, input, other, output);

      if (std::fabs(output_data[0] - 5.0f) > 1e-6f ||
          std::fabs(output_data[1] - 7.0f) > 1e-6f ||
          std::fabs(output_data[2] - 9.0f) > 1e-6f) {
        return 1;
      }

      output_data[0] = 0.0f;
      output_data[1] = 0.0f;
      output_data[2] = 0.0f;

      infini::ops::Add::Call(input, other, output);

      if (std::fabs(output_data[0] - 5.0f) > 1e-6f ||
          std::fabs(output_data[1] - 7.0f) > 1e-6f ||
          std::fabs(output_data[2] - 9.0f) > 1e-6f) {
        return 1;
      }

      return 0;
    }
    """
).lstrip()


_ADD_RETURN_SMOKE_SOURCE = textwrap.dedent(
    r"""
    #include <infini/ops.h>

    #include <cmath>
    #include <functional>
    #include <numeric>
    #include <stdexcept>
    #include <utility>
    #include <vector>

    class OwningTensor {
     public:
      using Shape = infini::ops::Tensor::Shape;
      using Strides = infini::ops::Tensor::Strides;

      OwningTensor(std::vector<float> data, Shape shape)
          : data_{std::move(data)},
            shape_{std::move(shape)},
            strides_{ContiguousStrides(shape_)},
            dtype_{infini::ops::DataType::kFloat32},
            device_{infini::ops::Device::Type::kCpu} {}

      static OwningTensor Empty(const Shape& shape, infini::ops::DataType dtype,
                                infini::ops::Device device) {
        if (dtype != infini::ops::DataType::kFloat32 ||
            device.type() != infini::ops::Device::Type::kCpu) {
          throw std::runtime_error("unexpected output metadata");
        }

        return OwningTensor(std::vector<float>(Numel(shape)), shape);
      }

      void* data() { return data_.data(); }

      const void* data() const { return data_.data(); }

      const Shape& shape() const { return shape_; }

      const Strides& strides() const { return strides_; }

      infini::ops::DataType dtype() const { return dtype_; }

      infini::ops::Device device() const { return device_; }

     private:
      static std::size_t Numel(const Shape& shape) {
        return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                               std::multiplies<std::size_t>());
      }

      static Strides ContiguousStrides(const Shape& shape) {
        if (shape.empty()) {
          return {};
        }

        Strides strides(shape.size());
        strides.back() = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2;
             i >= 0; --i) {
          strides[static_cast<std::size_t>(i)] =
              strides[static_cast<std::size_t>(i + 1)] *
              static_cast<infini::ops::Tensor::Stride>(
                  shape[static_cast<std::size_t>(i + 1)]);
        }
        return strides;
      }

      std::vector<float> data_;
      Shape shape_;
      Strides strides_;
      infini::ops::DataType dtype_;
      infini::ops::Device device_;
    };

    int main() {
      OwningTensor input({1.0f, 2.0f, 3.0f}, {3});
      OwningTensor other({4.0f, 5.0f, 6.0f}, {3});

      auto output = infini::ops::Add::Call(input, other);
      const auto* data = static_cast<const float*>(output.data());

      if (output.shape() != OwningTensor::Shape{3}) {
        return 1;
      }
      if (std::fabs(data[0] - 5.0f) > 1e-6f ||
          std::fabs(data[1] - 7.0f) > 1e-6f ||
          std::fabs(data[2] - 9.0f) > 1e-6f) {
        return 1;
      }

      OwningTensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
      OwningTensor b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});

      auto c = infini::ops::Gemm::Call(a, b);
      const auto* c_data = static_cast<const float*>(c.data());

      if (c.shape() != OwningTensor::Shape{2, 2}) {
        return 1;
      }
      if (std::fabs(c_data[0] - 58.0f) > 1e-6f ||
          std::fabs(c_data[1] - 64.0f) > 1e-6f ||
          std::fabs(c_data[2] - 139.0f) > 1e-6f ||
          std::fabs(c_data[3] - 154.0f) > 1e-6f) {
        return 1;
      }

      return 0;
    }
    """
).lstrip()
