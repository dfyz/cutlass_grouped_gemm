import os
from pathlib import Path
from setuptools import setup, find_packages


# Initialize ext_modules to an empty list
ext_modules = []
extra_deps = {}
cmdclass = {}

# Attempt to import torch and define CUDA extensions only if torch is available
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    cwd = Path(os.path.dirname(os.path.abspath(__file__)))
    _dc = torch.cuda.get_device_capability()
    _dc = f"{_dc[0]}{_dc[1]}"

    ext_modules = [
        CUDAExtension(
            "grouped_gemm_backend",
            ["csrc/ops.cu", "csrc/grouped_gemm.cu"],
            include_dirs=[
                f"{cwd}/third_party/cutlass/include/",
                f"{cwd}/csrc"
            ],
            extra_compile_args={
                "cxx": [
                    "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
                ],
                "nvcc": [
                    f"--generate-code=arch=compute_{_dc},code=sm_{_dc}",
                    f"-DGROUPED_GEMM_DEVICE_CAPABILITY={_dc}",
                    # NOTE: CUTLASS requires c++17.
                    "-std=c++17",
                ],
            }
        )
    ]
    # Define the build_ext command only if torch is available
    cmdclass = {"build_ext": BuildExtension}
except ImportError as e:
    print("PyTorch is not available, CUDA extensions will not be built.")


extra_deps['dev'] = [
    'absl-py',
]
extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)


setup(
    name="cutlass_grouped_gemm",
    author="One",
    author_email="imonenext@gmail.com",
    description="CUTLASS Grouped GEMM",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/imoneoi/cutlass_grouped_gemm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require=extra_deps,
)
