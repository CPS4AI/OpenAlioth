# TL;DR

```shell
git clone https://github.com/AntCPLab/OpenBumbleBee.git
cd OpenBumbleBee
python setup.py bdist_wheel

# modify psi/bazel/repositories.bzl line 129:143
# def _com_github_intel_ipp():
#     maybe(
#         http_archive,
#         name = "com_github_intel_ipp",
#         sha256 = "d70f42832337775edb022ca8ac1ac418f272e791ec147778ef7942aede414cdc",
#         strip_prefix = "cryptography-primitives-ippcp_2021.8",
#         build_file = "@psi//bazel:ipp.BUILD",
#         patch_args = ["-p1"],
#         patches = [
#             "@psi//bazel:patches/ippcp.patch",
#         ],
#         urls = [
#             "https://github.com/intel/ipp-crypto/archive/refs/tags/ippcp_2021.8.tar.gz",
#         ],
#     )

pip install dist/*.whl --force-reinstall
```

# WOE Example

1. Launch SPU backend runtime

    ```sh
    # OpenBumbleBee
    bazel run -c opt //examples/python/utils:nodectl --jobs 16 -- --config `pwd`/examples/python/conf/2pc_alioth.json up
    # alioth-impl
    python utils/nodectl.py --config `pwd`/conf/2pc_alioth.json up
    ```

2. Run `woe` example

    ```sh
    # OpenBumbleBee
    bazel run -c opt //examples/python/stats:woe --jobs 16 -- --config `pwd`/examples/python/conf/2pc_alioth.json
    # alioth-impl
    python woe.py -c `pwd`/conf/2pc_alioth.json
    ```

3. Exp

    ```
    python -u exp.py -s woe -m vp hp_cat hp_num -t 1 -H 1000 10000 100000 -W 10 50 100 -K 5 10 20 2>&1 | tee output/LANLOG
    python -u exp.py -s woe -m mbm_appqua -t 1 -H 1000 10000 100000 -W 1 2>&1 | tee output/LANLOG
    python -u exp.py -s woe -m mbm_transformation mbm_naive_transformation -t 1 -H 100000 -W 10 -K 5 10 20 2>&1 | tee output/LANLOG
    ```