# ALIOTH: An Efficient and Secure Weight-of-Evidence Framework for Privacy-Preserving Data Processing


## Introduction

ALIOTH is a two-party framework for WoE-based dataset transformation for horizontal and vertical data partitions.
This repository is built upon [SecretFlow-SPU](https://github.com/secretflow/spu) with optimizations from [OpenBumbleBee](https://github.com/AntCPLab/OpenBumbleBee), which provides a prototype implementation of ALIOTH, intended solely for experimental testing purposes and SHOULD NOT be used in production environments.



## Build

### TL;DR

```sh
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

## WOE Example

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

    ```sh
    python -u exp.py -s woe -m vp hp_cat hp_num -t 1 -H 1000 10000 100000 -W 10 50 100 -K 5 10 20 2>&1 | tee output/LANLOG
    # python -u exp.py -s woe -m mbm_appqua -t 1 -H 1000 10000 100000 -W 1 2>&1 | tee output/LANLOG
    python -u exp.py -s woe -m mbm_appqua -t 1 -H 10000 -W 1 -K 100 1000 10000 2>&1 | tee output/LANLOG
    python -u exp.py -s woe -m mbm_transformation mbm_naive_transformation -t 1 -H 10000 -W 100 -K 5 10 20 2>&1 | tee output/LANLOG
    ```
    ### GCD, VP
    ```sh
    python -u exp.py -s woe -m vp -t 1 -H 800 -W 20 -K 5 2>&1 | tee output/LANLOG_GCD_VP
    ```
    ### GCD, HP
    ```sh
    python -u exp.py -s woe -m hp_cat -t 1 -H 800 -W 13 -K 5 2>&1 | tee output/LANLOG_GCD_HP_CAT
    python -u exp.py -s woe -m hp_num -t 1 -H 800 -W 7 -K 5 2>&1 | tee output/LANLOG_GCD_HP_NUM
    ```
    # HCDR, VP
    ```sh
    python -u exp.py -s woe -m vp -t 1 -H 307511 -W 51 -K 5 2>&1 | tee output/LANLOG_HCDR_VP_CAT
    python -u exp.py -s woe -m vp -t 1 -H 307511 -W 69 -K 10 2>&1 | tee output/LANLOG_HCDR_VP_NUM
    ```
    # HCDR, HP
    ```sh
    python -u exp.py -s woe -m hp_cat -t 1 -H 307511 -W 51 -K 5 2>&1 | tee output/LANLOG_HCDR_HP_CAT
    python -u exp.py -s woe -m hp_num -t 1 -H 307511 -W 69 -K 10 2>&1 | tee output/LANLOG_HCDR_HP_NUM
    python -u exp.py -s woe -m iv_gcd_vp iv_gcd_hp iv_hcdr_vp iv_hcdr_hp 2>&1 | tee output/LAN_LOG_IV
    ```

    # HCDR, VP
    ```sh
    python -u exp.py -s woe -m vp -t 1 -H 307511 -W 69 -K 10 2>&1 | tee output/WANLOG_HCDR_VP_NUM
    ```
    # HCDR, HP
    ```sh
    python -u exp.py -s woe -m hp_num -t 1 -H 307511 -W 69 -K 10 2>&1 | tee output/WANLOG_HCDR_HP_NUM
    ```
    # IV
    ```sh
    python -u exp.py -s woe -m iv_gcd_vp iv_gcd_hp iv_hcdr_vp iv_hcdr_hp 2>&1 | tee output/WANLOG_IV
    ```
    # WOE HP_NUM
    ```sh
    python -u exp.py -s woe -m hp_num -t 1 -H 1000 10000 100000 -W 10 50 100 -K 5 10 20 2>&1 | tee output/WANLOG_HP_NUM
    ```
    # APPQUA
    ```sh
    python -u exp.py -s woe -m mbm_appqua -t 1 -H 1000 10000 100000 -W 1 2>&1 | tee output/WANLOG_APPQUA
    ```
    # LR regression training with WoE
    ```sh
    python lr.py -m gcd_woe | tee LOG_GCD_WOE
    python lr.py -m gcd_normal | tee LOG_GCD_NORMAL
    python lr.py -m hcdr_woe | tee LOG_HCDR_WOE
    python lr.py -m hcdr_normal | tee LOG_HCDR_NORMAL
    ```
