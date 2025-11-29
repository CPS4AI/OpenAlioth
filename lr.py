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


# FIXME: For un-normalized data, grad(sigmoid) is likely to overflow, either with exp/tanh or taylor series
# https://stackoverflow.com/questions/68290850/jax-autograd-of-a-sigmoid-always-returns-nan
def sigmoid(x):
    # return 0.5 * (jnp.tanh(x / 2) + 1)
    return 1 / (1 + jnp.exp(-x))


def predict(x, w, b):
    return sigmoid(jnp.matmul(x, w) + b)


def loss(x, y, w, b, use_cache):
    if use_cache:
        w = spu.experimental.make_cached_var(w)
        b = spu.experimental.make_cached_var(b)
    pred = predict(x, w, b)
    label_prob = pred * y + (1 - pred) * (1 - y)

    if use_cache:
        w = spu.experimental.drop_cached_var(w, label_prob)
        b = spu.experimental.drop_cached_var(b, label_prob)

    return -jnp.mean(jnp.log(label_prob))


class LogitRegression:
    def __init__(self, n_epochs=10, n_iters=10, step_size=0.1):
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.step_size = step_size

    def fit_auto_grad(self, feature, label, use_cache=False):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        if use_cache:
            feature = spu.experimental.make_cached_var(feature)

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for x, y in zip(xs, ys):
                grad = jax.grad(loss, argnums=(2, 3))(x, y, w_, b_, use_cache)
                w_ -= grad[0] * self.step_size
                b_ -= grad[1] * self.step_size

            return w_, b_

        ret = jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))

        if use_cache:
            feature = spu.experimental.drop_cached_var(feature, *ret)

        return ret

    def fit_manual_grad(self, feature, label, use_cache=False):
        w = jnp.zeros(feature.shape[1])
        b = 0.0

        if use_cache:
            feature = spu.experimental.make_cached_var(feature)

        xs = jnp.array_split(feature, self.n_iters, axis=0)
        ys = jnp.array_split(label, self.n_iters, axis=0)

        def body_fun(_, loop_carry):
            w_, b_ = loop_carry
            for x, y in zip(xs, ys):
                pred = predict(x, w_, b_)
                err = pred - y
                w_ -= jnp.matmul(jnp.transpose(x), err) / y.shape[0] * self.step_size
                b_ -= jnp.mean(err) * self.step_size

            return w_, b_

        ret = jax.lax.fori_loop(0, self.n_epochs, body_fun, (w, b))

        if use_cache:
            feature = spu.experimental.drop_cached_var(feature, *ret)

        return ret


def run_on_cpu(x_train, y_train):
    lr = LogitRegression()

    # w, b = jax.jit(lr.fit_auto_grad)(x_train, y_train)
    # print(w0, b0)

    w, b = jax.jit(lr.fit_manual_grad)(x_train, y_train)

    # return [w0, w1], [b0, b1]
    return w, b


SPU_OBJECT_META_PATH = "/tmp/driver_spu_jax_lr_object.txt"

import cloudpickle as pickle


def save_and_load_model(x_test, y_test, W, b):
    # 1. save metadata and spu objects.
    meta = ppd.save((W, b))
    with open(SPU_OBJECT_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    # 2. load metadata and spu objects.
    with open(SPU_OBJECT_META_PATH, "rb") as f:
        meta_ = pickle.load(f)
    W_, b_ = ppd.load(meta_)

    W_r, b_r = ppd.get(W_), ppd.get(b_)
    print(W_r, b_r)

    score = metrics.roc_auc_score(y_test, predict(x_test, W_r, b_r))
    print("AUC(save_and_load_model)={}".format(score))

    return score


def compute_score(x_test, y_test, W_r, b_r, type):
    y_pred_prob = predict(x_test, W_r, b_r)
    y_pred = y_pred_prob > 0.5
    print(f"LR {type} report:")
    print("+ Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("+ Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    score = metrics.roc_auc_score(y_test, y_pred_prob)
    print("AUC({})={}".format(type, score))
    return score
    


def run_on_spu(x, y, use_cache=False):
    def train(x1, x2, y):
        x = jnp.concatenate((x1, x2), axis=1)
        lr = LogitRegression()
        return lr.fit_manual_grad(x, y, use_cache)

    x1 = ppd.device("P1")(lambda x: x[:, :50])(x)
    x2 = ppd.device("P2")(lambda x: x[:, 50:])(x)
    y = ppd.device("P1")(lambda x: x)(y)
    W, b = ppd.device("SPU")(train)(x1, x2, y)

    return W, b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='distributed driver.')
    parser.add_argument(
        "-c", "--config", default="conf/2pc_alioth.json"
    )
    parser.add_argument(
        "-m", "--mode", default="hcdr_woe"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    ppd.init(conf["nodes"], conf["devices"])
    
    def load_dataset(mode):
        if args.mode == "hcdr_woe":
            return dsutil.hcdr_dataset_woe()
        elif args.mode == "hcdr_normal":
            return dsutil.hcdr_dataset_normal()
        elif args.mode == "gcd_woe":
            return dsutil.gcd_dataset_woe()
        elif args.mode == "gcd_normal":
            return dsutil.gcd_dataset_normal()
        else:
            raise NotImplementedError

    x_train, x_test, y_train, y_test = load_dataset(args.mode)
    if args.mode != "gcd_normal":
        x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    print('Run on CPU\n------\n')
    # w, b = run_on_cpu(x_train, y_train)
    # compute_score(x_test, y_test, w, b, 'cpu, manual_grad')
    # print('Run on SPU\n------\n')
    w, b = run_on_spu(x_train, y_train)
    w_r, b_r = ppd.get(w), ppd.get(b)
    compute_score(x_test, y_test, w_r, b_r, 'spu')
    # save_and_load_model(x, y, w, b)
    