# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > python examples/python/utils/nodectl.py -c examples/python/conf/2pc_semi2k.json up
#
# Run this example script.
# > python examples/python/ml/jax_lr/jax_lr.py


import argparse
import json

import jax
import jax.numpy as jnp
from sklearn import metrics

import dsutil
import spu.utils.distributed as ppd
import spu

import os

if "http_proxy" in os.environ:
    del os.environ["http_proxy"]
if "https_proxy" in os.environ:
    del os.environ["https_proxy"]

# "FLAGS_stack_size_normal=8388608 FLAGS_bthread_concurrency=8"
os.environ["FLAGS_stack_size_normal"] = "8388608"
os.environ["FLAGS_bthread_concurrency"] = "8"
env = os.environ.copy()

def run_on_spu():
    shape = (100000, 100, 20)
    x = jnp.zeros(shape)
    y = jnp.ones(shape)
    x = ppd.device("P1")(lambda x: x)(x)
    y = ppd.device("P2")(lambda x: x)(y)
    res = ppd.device("SPU")(lambda x, y: x < y)(x, y)
    res = ppd.get(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument(
        "-c", "--config", default="conf/2pc_alioth.json"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"])

    run_on_spu()
    