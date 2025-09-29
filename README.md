# TL;DR

```shell
# modify setup.py: disable all the psi dependencies: line 102, 210, 211, 213, 214
# modify WORKSPACE: disable psi_deps: line 30, 32
# modify WORKSPACE: line 91 ~ 111
python setup.py bdist_wheel

# modify psi/bazel/repositories.bzl line 129:143
def _com_github_intel_ipp():
    maybe(
        http_archive,
        name = "com_github_intel_ipp",
        sha256 = "d70f42832337775edb022ca8ac1ac418f272e791ec147778ef7942aede414cdc",
        strip_prefix = "cryptography-primitives-ippcp_2021.8",
        build_file = "@psi//bazel:ipp.BUILD",
        patch_args = ["-p1"],
        patches = [
            "@psi//bazel:patches/ippcp.patch",
        ],
        urls = [
            "https://github.com/intel/ipp-crypto/archive/refs/tags/ippcp_2021.8.tar.gz",
        ],
    )

pip install dist/*.whl --force-reinstall
```

# WOE Example

1. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl --jobs 16 -- --config `pwd`/examples/python/conf/2pc.json up
    python
    ```

23. Run `flax_gpt2` example

    ```sh
    bazel run -c opt //examples/python/stats:woe --jobs 16 -- --config `pwd`/examples/python/conf/2pc.json
    ```
