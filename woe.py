import argparse
import json

import jax
import jax.numpy as jnp
# from jax.nn import identity
from jax import random
import spu
import time
import argparse

# exp_H           = [1000, 10000, 100000]    # 1K, 10K, 100K
# exp_W           = [10, 50, 100]
# exp_K           = [5, 10, 20]

def identity(x):
    return x

'''
Quantile.
'''
def percentile(u: jax.Array, v: jax.Array, beta: jax.Array):
    x = jnp.vstack((u, v))
    # compute the quantile via jax.numpy
    return jnp.quantile(x, q=beta, axis=0).transpose((1, 0))

class DDSketch:
    alpha = None
    beta = None
    gamma = None
    K = None
    bucket_offset = None
    
    def __init__(self, alpha: float, beta: jax.Array, K: int = None, bucket_offset: jax.Array = jnp.array(0)):
        self.alpha = alpha
        self.gamma = (1 + alpha) / (1 - alpha)
        self.beta = beta if len(beta.shape) == 2 else beta[jnp.newaxis, :]  # (W, quantile_counts)
        self.K = K
        self.bucket_offset = bucket_offset
        
    def __repr__(self):
        return f"DDSketch(alpha={self.alpha}, gamma={self.gamma}, beta={self.beta}, K={self.K}, bucket_offset={self.bucket_offset})"
        
    '''
    Implementation of DDSketch algorithm and AppQua protocol, which computes an approximate quantile.
    @param:
        u: 2d-array which represents P0's input. (n_samples, n_features), i.e. (H, W).
        v: 2d-array which represents P1's input. (n_samples, n_features), i.e. (H, W).
        beta: 2d-array which represents quantiles, i.e. (W, quantile_counts).
        alpha: relative error of DDSketch.
        K: the number of buckets. Estimated value.
        bucket_offset: the **lowest** absolute logical bucket index. Estimated value. (n_buckets)
    '''
    def ddsketch(self, u: jax.Array, v: jax.Array) -> jax.Array:
        # AppQua step 1-3
        # compute the samples on the quantile
        print("----------------------------------------------------\nstep 1-3 begins")
        quantile_counts = self.beta.shape[1]
        (H1, W), H2 = u.shape, v.shape[0]
        k = jnp.floor(self.beta * (H1 + H2)).astype(jnp.int32)
        if k.size != W * quantile_counts:
            k = jnp.repeat(k, W, axis=0)    # (W, quantile_counts)
            assert k.size == W * quantile_counts
        print(f"params:\n\tddsketch: {self},\n\tH1: {H1}, H2: {H2}, W: {W},\n\tquantile_counts: {quantile_counts},\n\tk: {k}")
        print("----------------------------------------------------\nstep 1-3 ends")
        
        # AppQua step 4-9
        # compute how many samples are in each bucket
        # share s_A, s_B here
        print("----------------------------------------------------\nstep 4-9 begins")
        s_A = self.insert_into_buckets(u)
        s_B = self.insert_into_buckets(v)
        s = s_A + s_B
        s = jnp.cumsum(s, axis=1)   # (W, 2K+1)
        print("----------------------------------------------------\nstep 4-9 ends")
        
        # AppQua step 10-14
        # compute the upper bound of each bucket
        print("----------------------------------------------------\nstep 10-14 begins")
        tau = jnp.arange(-self.K, self.K + 1)   # (2K+1)
        tau = jnp.tile(tau, (W, 1)) # (W, 2K+1)
        tau = tau - self.bucket_offset[:, jnp.newaxis]
        tau = tau.at[:, self.K].set(0)
        coeff = jnp.zeros((2 * self.K + 1))
        coeff = coeff.at[:self.K].set(-2 / (self.gamma + 1))
        coeff = coeff.at[self.K+1:].set(2 / (self.gamma + 1))
        q = jnp.full((W, 2 * self.K + 1), self.gamma)   # (W, 2K+1)
        q = q ** tau * coeff  # (W, 2K+1)
        print("----------------------------------------------------\nstep 10-14 ends")
        
        # AppQua step 15-19
        print("----------------------------------------------------\nstep 15-19 begins")
        a = jnp.full((W, quantile_counts, 2 * self.K + 2), False)   # (W, quantile_counts, 2K+2)
        a = a.at[:, :, 1:].set(k[:, :, jnp.newaxis] < s[:, jnp.newaxis, :])
        a = a[:, :, :-1] ^ a[:, :, 1:]   # (W, quantile_counts, 2K+1)
        print("----------------------------------------------------\nstep 15-19 ends")
        # return jnp.einsum('bmn,bn->bm', a, q) 
        return jax.lax.batch_matmul(a, q[:,:,jnp.newaxis]).squeeze(-1)
    
    def ddsketch_local(self, X: jax.Array):
        return self.insert_into_buckets(X)
    
    def ddsketch_global(self, Su: jax.Array, Sv: jax.Array, W: int):
        # AppQua step 1-3
        # compute the samples on the quantile
        quantile_counts = self.beta.shape[1]
        H1, H2 = Su.shape[0], Sv.shape[0]
        k = jnp.floor(self.beta * (H1 + H2)).astype(jnp.int32)
        if k.size != W * quantile_counts:
            k = jnp.repeat(k, W, axis=0)    # (W, quantile_counts)
            
        # AppQua step 4-9
        # compute how many samples are in each bucket
        # share s_A, s_B here
        s = Su + Sv
        s = jnp.cumsum(s, axis=1)   # (W, 2K+1)
        
        # AppQua step 10-14
        # compute the upper bound of each bucket
        print(f"self.K: {self.K}, self.bucket_offset: {self.bucket_offset}")
        tau = jnp.arange(-self.K, self.K + 1)   # (2K+1)
        tau = jnp.tile(tau, (W, 1)) # (W, 2K+1)
        tau = tau - self.bucket_offset[:, jnp.newaxis]
        tau = tau.at[:, self.K].set(0)
        coeff = jnp.zeros((2 * self.K + 1))
        coeff = coeff.at[:self.K].set(-2 / (self.gamma + 1))
        coeff = coeff.at[self.K+1:].set(2 / (self.gamma + 1))
        q = jnp.full((W, 2 * self.K + 1), self.gamma)   # (W, 2K+1)
        q = q ** tau * coeff  # (W, 2K+1)
        
        # AppQua step 15-19
        a = jnp.full((W, quantile_counts, 2 * self.K + 2), False)   # (W, quantile_counts, 2K+2)
        a = a.at[:, :, 1:].set(k[:, :, jnp.newaxis] < s[:, jnp.newaxis, :])
        a = a[:, :, :-1] ^ a[:, :, 1:]   # (W, quantile_counts, 2K+1)
        return jax.lax.batch_matmul(a, q[:,:,jnp.newaxis]).squeeze(-1)
        
    def logical_index(self, X: jax.Array):
        return jnp.ceil(jnp.log(X) / jnp.log(self.gamma)).astype(jnp.int32)
    
    def insert_into_buckets(self, X: jax.Array):
        (H, W) = X.shape
        X_neg, X_pos = X[X < 0], X[X > 0]
        col_neg, col_zero, col_pos = jnp.where(X < 0)[1], jnp.where(X == 0)[1],jnp.where(X > 0)[1]
        
        # on negative input
        B_neg = jnp.zeros((W, self.K))
        if X_neg.size > 0:
            tau = self.logical_index(-X_neg) + self.bucket_offset[col_neg] + self.K * col_neg
            temp = jnp.bincount(tau, minlength=exp_OVERFLOW_FACTOR * W * self.K).reshape((W, -1))
            if temp.shape[1] > self.K * exp_OVERFLOW_FACTOR:
                
                raise Exception(f"Overflowed positive value: {temp.shape[1]} vs {self.K} x {exp_OVERFLOW_FACTOR}")
            sum_overflow = jnp.sum(temp[:, self.K:], axis=1)
            if sum_overflow.size > 0:
                print( "Warning: bucket numbers (K) is too small to hold all negative values.")
                print(f"Given K: {self.K}, # of bin count K: {temp.shape[1]}")
                print( "The overflowed negative values are merged into the K-th bucket.")
                temp = temp.at[:, self.K - 1].set(temp[:, self.K - 1] + sum_overflow)
                temp = temp[:, :self.K]
            B_neg = temp.reshape((W, self.K))
        
        # on zero input
        B_zero = jnp.bincount(col_zero, minlength=W).reshape(W, 1)
        
        # on positive input
        B_pos = jnp.zeros((W, self.K))
        if X_pos.size > 0:
            tau = self.logical_index(X_pos) + self.bucket_offset[col_pos] + self.K * col_pos
            temp = jnp.bincount(tau, minlength=exp_OVERFLOW_FACTOR * W * self.K).reshape((W, -1))
            if temp.shape[1] > self.K * exp_OVERFLOW_FACTOR:
                raise Exception(f"Overflowed positive value: {temp.shape[1]} vs {self.K} x {exp_OVERFLOW_FACTOR}")
            sum_overflow = jnp.sum(temp[:, self.K:], axis=1)
            if sum_overflow.size > 0:
                print( "Warning: bucket numbers (K) is too small to hold all positive values.")
                print(f"Given K: {self.K}, # of bin count K: {temp.shape[1]}")
                print( "The overflowed positive values are merged into the K-th bucket.")
                temp = temp.at[:, self.K - 1].set(temp[:, self.K - 1] + sum_overflow)
                temp = temp[:, :self.K]
            B_pos = temp.reshape((W, self.K))
        
        return jnp.hstack((B_neg[:, ::-1], B_zero, B_pos))
        
    '''
    Helper methods to specify hyper parameters.
    '''
    def update_params_from_range(self, min_abs_value: jax.Array, max_abs_value: jax.Array):
        min_logical_index = self.logical_index(min_abs_value)
        max_logical_index = self.logical_index(max_abs_value)
        self.bucket_offset = -min_logical_index
        self.K = jnp.max(max_logical_index - min_logical_index + 1).astype(jnp.int32).item()
        return self.K, self.bucket_offset
    
    def get_bucket_offset(self, min_abs_value: jax.Array, max_abs_value: jax.Array):
        min_logical_index = self.logical_index(min_abs_value)
        return -min_logical_index
    
    def abs_upper_bound(self, logical_index: jax.Array):
        return 2 * self.gamma ** logical_index / (self.gamma + 1)
        
    def abs_lower_bound(self, logical_index: jax.Array):
        return 2 * self.gamma ** (logical_index - 1) / (self.gamma + 1)
    
    
def vectorized_sec_woe(z_pos: jax.Array, z_neg: jax.Array, zeta: jax.Array):
    divz = z_pos / z_neg
    
    temp = divz * zeta
    
    return jnp.log(temp)


def transformation(B: jax.Array, w: jax.Array, max_cat: int = 1):
    # ! Alert: W' = W * max_cat, i.e. B and w has been padding.
    # B: (H, W')
    # w: (W', num_labels)
    B = jnp.reshape(B, (B.shape[0], B.shape[1] // max_cat, max_cat))  # B: (H, W, max_cat)
    B = jnp.transpose(B, (1, 0, 2)) # B: (W, H, max_cat)
    w = jnp.reshape(w, (w.shape[0] // max_cat, max_cat, w.shape[1]))  # w: (W, max_cat, num_labels)
    Lam = jax.lax.batch_matmul(B, w)
    Lam = jnp.transpose(Lam, (1, 0, 2))
    return Lam

def transformation_with_max_cat(max_cat):
    m_max_cat = max_cat
    def transformation_inner(B, w):
        return transformation(B, w, max_cat=m_max_cat)
    return transformation_inner


'''
WOE for vertical partition.
'''
class WoeVp():
    # P1 computes locally.
    def get_zeta(self, y: jax.Array, H: int):
        num_1_label = jnp.count_nonzero(y, axis=0)
        num_0_label = H - num_1_label
        
        zeta = num_0_label / num_1_label
        
        return zeta

    # P1 computes locally or P0 computes with P1.
    # B: (H, W)
    # y: (H, num_labels)
    # output: z_pos (W, num_labels), z_neg (W, num_labels)
    def get_z(self, B: jax.Array, y: jax.Array):
        z_pos = B.T @ y # (W, num_labels)
        sum_B = jnp.sum(B, axis=0)  # (W)
        z_neg = sum_B[:, jnp.newaxis] - z_pos # (W, num_labels)
        
        return z_pos, z_neg
    
    
'''
WOE for horizontal partition.
'''
class WoeHp():
    # HP cat/num, seperately
    # no comm in cat scene
    def get_z(self, Bu: jax.Array, Bv: jax.Array, yu: jax.Array, yv: jax.Array):
        z_pos = Bu.T @ yu + Bv.T @ yv # (W, num_labels)
        sum_B = jnp.sum(Bu, axis=0) + jnp.sum(Bv, axis=0) # (W)
        z_neg = sum_B[:, jnp.newaxis] - z_pos # (W, num_labels)
        return z_pos, z_neg
    
    # # HP
    # def get_z_pos(self, Bu: jax.Array, yu: jax.Array, Bv: jax.Array, yv: jax.Array):
    #     z_pos = Bu @ yu + Bv @ yv # (W, num_labels)
        
    #     return z_pos
    
    # def get_sum_B(self, B: jax.Array):
    #     sum_B = jnp.sum(B, axis=0)  # (W)
        
    #     return sum_B
    
    # def get_z_neg(self, sum_Bu: jax.Array, sum_Bv: jax.Array, z_pos: jax.Array):
    #     sum_B = sum_Bu + sum_Bv
    #     z_neg = sum_B - z_pos # (W, num_labels)
        
    #     return z_neg
    
    # def count_label(self, H: int, y: jax.Array):
    #     num_1_label = jnp.count_nonzero(y, axis=0)
    #     num_0_label = H - num_1_label
        
    #     return num_1_label, num_0_label
    
    # def get_zeta(self, num_0_label_u: jax.Array, num_0_label_v: jax.Array, num_1_label_u: jax.Array, num_1_label_v: jax.Array):
    #     num_1_label = num_1_label_u + num_1_label_v
    #     num_0_label = num_0_label_u + num_0_label_v
        
    #     zeta = num_0_label / num_1_label
        
    #     return zeta
    
    def get_zeta(self, Hu: jax.Array, Hv: jax.Array, yu: jax.Array, yv: jax.Array):
        num_1_label_u = jnp.count_nonzero(yu, axis=0)
        num_0_label_u = Hu - num_1_label_u
        
        num_1_label_v = jnp.count_nonzero(yv, axis=0)
        num_0_label_v = Hv - num_1_label_v
        
        num_1_label = num_1_label_u + num_1_label_v
        num_0_label = num_0_label_u + num_0_label_v
        
        zeta = num_0_label / num_1_label
        
        return zeta
    
    def get_all_for_num(self, B: jax.Array, yu: jax.Array, yv: jax.Array):
        H, W = B.shape
        y = jnp.vstack((yu, yv))
        z_pos = B.T @ y # (W, num_labels)
        sum_B = jnp.sum(B, axis=0)  # (W)
        z_neg = sum_B[:, jnp.newaxis] - z_pos # (W, num_labels)
        
        num_1_label_u = jnp.count_nonzero(yu, axis=0)
        num_1_label_v = jnp.count_nonzero(yv, axis=0)
        num_1_label   = num_1_label_u + num_1_label_v
        num_0_label = H - num_1_label
        zeta = num_0_label / num_1_label
        
        return z_pos, z_neg, zeta


'''
Dataset.
'''
class Synthesis():
    H, W = None, None
    num_labels, max_cats = None, None
    
    def __init__(self, H, W, num_labels, max_cats):
        self.H, self.W = H, W
        self.num_labels, self.max_cats = num_labels, max_cats
        
    def generate_dataset(self):
        X = random.randint(random.PRNGKey(0), (self.H, self.W), minval=0, maxval=self.max_cats)
        y = random.randint(random.PRNGKey(1), (self.H, self.num_labels), minval=0, maxval=2)
        return X, y
    
    def generate_num_dataset(self):
        X = random.uniform(random.PRNGKey(0), (self.H, self.W), minval=exp_MIN_NUM_VAL, maxval=exp_MAX_NUM_VAL)
        y = random.randint(random.PRNGKey(1), (self.H, self.num_labels), minval=0, maxval=2)
        return X, y
    
    def split_vp(self, X, y, ratio):
        n = int(self.W * ratio)
        return X[:, :n], X[:, n:], y
    
    def split_hp(self, X, y, ratio):
        n = int(self.H * ratio)
        return X[:n, :], X[n:, :], y[:n, :], y[n:, :]
    

'''
Data Encoder.
'''
class DataEncoder():
    def __init__(self):
        pass
    
    def encode_vp(self, X: jax.Array):
        H, W = X.shape
        
        max_cats = jnp.max(X, axis=0) + 1
        binary_features = []
        
        for feature_idx in range(W):
            n_classes = max_cats[feature_idx]
            
            for class_val in range(n_classes):
                binary_feature = (X[:, feature_idx] == class_val).astype(jnp.int32)
                binary_features.append(binary_feature)
        
        return jnp.vstack(binary_features).T # (H, W = W * n_classes)
    
    def encode_hp_cat(self, X: jax.Array, max_cats: jax.Array):
        H, W = X.shape
        
        binary_features = []
        
        for feature_idx in range(W):
            n_classes = max_cats[feature_idx]
            
            for class_val in range(n_classes):
                binary_feature = (X[:, feature_idx] == class_val).astype(jnp.int32)
                binary_features.append(binary_feature)

        res = jnp.vstack(binary_features).T
        print(f"encode_hp_cat: from ({H}, {W}) to {res.shape}")

        return res

    def encode_hp_num(self, I: jax.Array, U: jax.Array, V: jax.Array, alpha, beta, K, buckets_offset):
        quantile_counts = I.shape[1]
        
        # share U, V here
        X = jnp.vstack((U, V))  # (H, W)
        H, W = X.shape
        C = jnp.full((H, W, quantile_counts + 1), True) # (H, W, quantile_counts + 1)
        print(f"encode_hp_num: {X.shape}, {I.shape}")
        if exp_FLAG_LARGE_DATASET:
            print(f"encode_hp_num: too many elements to perform LessThan.")
            C = C.at[:H//4, :, :-1].set(X[:H//4, :, jnp.newaxis] < I[jnp.newaxis, :, :])
            C = C.at[H//4:H//2, :, :-1].set(X[H//4:H//2, :, jnp.newaxis] < I[jnp.newaxis, :, :])
            C = C.at[H//2:H*3//4, :, :-1].set(X[H//2:H*3//4, :, jnp.newaxis] < I[jnp.newaxis, :, :])
            C = C.at[H*3//4:, :, :-1].set(X[H*3//4:, :, jnp.newaxis] < I[jnp.newaxis, :, :])
        else:
            C = C.at[:, :, :-1].set(X[:, :, jnp.newaxis] < I[jnp.newaxis, :, :])
        C = C.at[:, :, :-1].set(C[:, :, :-1] ^ C[:, :, 1:])
        binary_features = C  # (H, W, quantile_counts + 1)
        
        return binary_features.reshape(H, W * (quantile_counts + 1))
    

def compute_iv(B, label, woe_value):
    
        pos_feature_count = jnp.sum(B, axis=0)
        pos_label_count = jnp.sum(label, axis=0)
        neg_label_count = label.shape[0] - pos_label_count
        pos_feature_and_pos_label_count = label[jnp.newaxis, :] @ B
        pos_feature_and_neg_label_count = pos_feature_count - pos_feature_and_pos_label_count
        pos_label_rate = pos_feature_and_pos_label_count / pos_label_count
        neg_label_rate = pos_feature_and_neg_label_count / neg_label_count
        iv = (pos_label_rate - neg_label_rate) * woe_value
        
        return iv
    

import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

def p0_input(x):
    x = ppd.device("P1")(identity)(x)
    return ppd.device("SPU")(identity)(x)
    
def p1_input(x):
    x = ppd.device("P2")(identity)(x)
    return ppd.device("SPU")(identity)(x)

def co_input(x):
    s0 = jnp.zeros_like(x)
    s1 = x - s0
    s0 = ppd.device("P1")(identity)(s0)
    s1 = ppd.device("P2")(identity)(s1)
    return ppd.device("SPU")(jnp.add)(s0, s1)
    

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-m", "--mode", default="vp", help="vp or hp_num or hp_cat")
parser.add_argument("-c", "--config", default="conf/2pc.json")
parser.add_argument("-H", default=1000, type=int, help="number of samples")
parser.add_argument("-W", default=50, type=int, help="num of features")
parser.add_argument("-K", default=10, type=int, help="num of classes")
parser.add_argument("-a", "--alpha", default=0.01, type=float, help="error rate of DDSketch")
parser.add_argument("-b", "--beta", nargs="+", default=[0.5], type=float, help="list of partition ratio")
parser.add_argument("-t", "--times", default=1, type=int, help="number of experiments")
args = parser.parse_args()

exp_MODE        = args.mode
exp_TIMES       = args.times
exp_H           = args.H
exp_W           = args.W
exp_K           = args.K
exp_MIN_NUM_VAL = 1
exp_MAX_NUM_VAL = 4
exp_ALPHA       = args.alpha
exp_BETA        = args.beta
exp_OVERFLOW_FACTOR = 5
exp_FLAG_LARGE_DATASET = False

with open(args.config, 'r') as file:
    conf = json.load(file)
    
ppd.init(conf["nodes"], conf["devices"])


# from `get_z` to `vectorized_sec_woe`
def exp_vp(H, W, K, exp_times=1):
    print("-----------------------VP setting-----------------------")
    
    print("+ generate dataset")
    dataset_generator = Synthesis(H, W, 1, K)
    X, y = dataset_generator.generate_dataset()
    U, V, y = dataset_generator.split_vp(X, y, 0.5)
    W1, W2 = U.shape[1], V.shape[1]

    total_start_time = time.time()
    for i in range(exp_times):
        print("+ exp iteration {}".format(i))
        start_time = time.time()
        # print("+ encode vp")
        data_encoder = DataEncoder()
        BU = ppd.device("P1")(data_encoder.encode_vp)(U)
        BV = ppd.device("P2")(data_encoder.encode_vp)(V)

        # print("+ compute woe")
        woevp = WoeVp()
        zeta = ppd.device("P2")(woevp.get_zeta)(y, H)
        z_pos_bob, z_neg_bob = ppd.device("P2")(woevp.get_z)(BV, y)

        y = ppd.device("P2")(lambda x: x)(y)
        z_pos_alice, z_neg_alice = ppd.device("SPU")(woevp.get_z)(BU, y)

        z_pos, z_neg = ppd.device("SPU")(lambda xp, xn, yp, yn: (jnp.vstack((xp, yp)), jnp.vstack((xn, yn))))(z_pos_alice, z_neg_alice, z_pos_bob, z_neg_bob)
        w = ppd.device("SPU")(vectorized_sec_woe)(z_pos, z_neg, zeta)
        B = ppd.device("SPU")(lambda BU, BV: jnp.hstack((BU, BV)))(BU, BV)
        res = ppd.device("SPU")(transformation_with_max_cat(K))(B, w)
        res = ppd.get(res)
        print("- woe vp time: {:.2f} s".format(time.time() - start_time))
    print("- total time: {:.2f} s".format(time.time() - total_start_time))
    

# from `get_zeta` to `vectorized_sec_woe`
def exp_hp_cat(H, W, K, exp_times=1):
    print("-----------------------HP setting, cat-----------------------")
    
    print("+ generate dataset")
    dataset_generator = Synthesis(H, W, 1, K)
    X, y = dataset_generator.generate_dataset()
    U, V, yu, yv = dataset_generator.split_hp(X, y, 0.5)
    H1, H2 = U.shape[0], V.shape[0]
    max_cats = jnp.max(X, axis=0) + 1

    total_start_time = time.time()
    for i in range(exp_times):
        print("+ exp iteration {}".format(i))
        start_time = time.time()
        data_encoder = DataEncoder()
        BU = ppd.device("P1")(data_encoder.encode_hp_cat)(U, max_cats)
        BV = ppd.device("P2")(data_encoder.encode_hp_cat)(V, max_cats)
        yu = ppd.device("P1")(lambda x: x)(yu)
        yv = ppd.device("P2")(lambda x: x)(yv)

        woehp = WoeHp()
        zeta = ppd.device("SPU")(woehp.get_zeta)(H1, H2, yu, yv)
        z_pos, z_neg = ppd.device("SPU")(woehp.get_z)(BU, BV, yu, yv)

        w = ppd.device("SPU")(vectorized_sec_woe)(z_pos, z_neg, zeta)
        B = ppd.device("SPU")(lambda BU, BV: jnp.vstack((BU, BV)))(BU, BV)
        res = ppd.device("SPU")(transformation_with_max_cat(K))(B, w)
        res = ppd.get(res)
        print("- woe hp cat time: {:.2f} s".format(time.time() - start_time))
    print("- total time: {:.2f} s".format(time.time() - total_start_time))


# from `ddsketch_global` to `vectorized_sec_woe`
def exp_hp_num(H, W, K, alpha, beta, exp_times=1):
    print("-----------------------HP setting, num-----------------------")
    
    print("+ generate dataset")
    dataset_generator = Synthesis(H, W, 1, K)
    X, y = dataset_generator.generate_num_dataset()
    U, V, yu, yv = dataset_generator.split_hp(X, y, 0.5)
    H1, H2 = U.shape[0], V.shape[0]
    dds = DDSketch(alpha, beta)
    dds.update_params_from_range(jnp.min(X, axis=0), jnp.max(X, axis=0))
    # dds.bucket_offset = dds.get_bucket_offset(jnp.min(X, axis=0), jnp.max(X, axis=0))
    # dds.K = K
    
    def ddsketch_global(Su, Sv):
        return dds.ddsketch_global(Su, Sv, W)
    
    def encode_hp_num(I, U, V):
        return data_encoder.encode_hp_num(I, U, V, alpha, beta, K, dds.bucket_offset)
    
    total_start_time = time.time()
    for i in range(exp_times):
        print("+ exp iteration {}".format(i))
        U, V, yu, yv = dataset_generator.split_hp(X, y, 0.5)
        start_time = time.time()
        U = ppd.device("P1")(lambda x: x)(U)
        V = ppd.device("P2")(lambda x: x)(V)
        yu = ppd.device("P1")(lambda x: x)(yu)
        yv = ppd.device("P2")(lambda x: x)(yv)

        data_encoder = DataEncoder()
        Su = ppd.device("P1")(dds.ddsketch_local)(U)
        Sv = ppd.device("P2")(dds.ddsketch_local)(V)
        I = ppd.device("SPU")(ddsketch_global)(Su, Sv)
        B = ppd.device("SPU")(encode_hp_num)(I, U, V)

        woehp = WoeHp()
        z_pos, z_neg, zeta = ppd.device("SPU")(woehp.get_all_for_num)(B, yu, yv)

        w = ppd.device("SPU")(vectorized_sec_woe)(z_pos, z_neg, zeta)
        res = ppd.device("SPU")(transformation_with_max_cat(len(beta) + 1))(B, w)
        res = ppd.get(res)
        print("- woe hp num time: {:.2f} s".format(time.time() - start_time))
    print("- total time: {:.2f} s".format(time.time() - total_start_time))
    
    
def exp_mbm_appqua(H, W, K, alpha, beta, exp_times=1):
    print("-----------------------HP setting, num-----------------------")
    
    print("+ generate dataset")
    dataset_generator = Synthesis(H, W, 1, K)
    X, y = dataset_generator.generate_num_dataset()
    U, V, yu, yv = dataset_generator.split_hp(X, y, 0.5)
    H1, H2 = U.shape[0], V.shape[0]
    dds = DDSketch(alpha, beta)
    dds.update_params_from_range(jnp.min(X, axis=0), jnp.max(X, axis=0))
    dds.K = K
    # dds.K = K
    # dds.bucket_offset = dds.get_bucket_offset(jnp.min(X, axis=0), jnp.max(X, axis=0))
    
    def ddsketch_global(Su, Sv):
        return dds.ddsketch_global(Su, Sv, W)
    
    total_start_time = time.time()
    for i in range(exp_times):
        print("+ exp iteration {}".format(i))
        U, V, yu, yv = dataset_generator.split_hp(X, y, 0.5)
        start_time = time.time()
        U = ppd.device("P1")(lambda x: x)(U)
        V = ppd.device("P2")(lambda x: x)(V)
        yu = ppd.device("P1")(lambda x: x)(yu)
        yv = ppd.device("P2")(lambda x: x)(yv)

        data_encoder = DataEncoder()
        Su = ppd.device("P1")(dds.ddsketch_local)(U)
        Sv = ppd.device("P2")(dds.ddsketch_local)(V)
        I = ppd.device("SPU")(ddsketch_global)(Su, Sv)

        print("- microbenchmark appqua time: {:.2f} s".format(time.time() - start_time))
    print("- total time: {:.2f} s".format(time.time() - total_start_time))
    

def exp_mbm_transformation(Kj, Km, H=10000, W=10, num_label=1, exp_times=1):
    print("-----------------------transformation-----------------------")
    
    W = W * Km
    start_time = time.time()
    for i in range(exp_times):
        w0, w1 = jnp.zeros((W, num_label), dtype=jnp.float32), jnp.ones((W, num_label), dtype=jnp.float32)
        B0, B1 = jnp.zeros((H, W), dtype=jnp.int32), jnp.ones((H, W), dtype=jnp.int32)
        
        B0, B1 = ppd.device("P1")(lambda x: x)(B0), ppd.device("P2")(lambda x: x)(B1)
        w0, w1 = ppd.device("P1")(lambda x: x)(w0), ppd.device("P2")(lambda x: x)(w1)
        B = ppd.device("SPU")(lambda B0, B1: B0 + B1)(B0, B1)
        w = ppd.device("SPU")(lambda w0, w1: w0 + w1)(w0, w1)
        
        transformation_start = time.time()
        res = ppd.device("SPU")(transformation_with_max_cat(Km))(B, w)
        res = ppd.get(res)
        print("- microbenchmark transformation time: {:.2f} s".format(time.time() - transformation_start))
    print("- total time: {:.2f} s".format(time.time() - start_time))
    

def exp_mbm_naive_transformation(Kj, Km, H=10000, W=100, num_label=1, exp_times=1):
    print("-----------------------naive transformation-----------------------")
    
    def naive_transformation(B, w):
        return jnp.matmul(B, w)
    
    start_time = time.time()
    for i in range(exp_times):
        w0, w1 = [p0_input(jnp.zeros((Kj, num_label), dtype=jnp.float32))], [p1_input(jnp.ones((Kj, num_label), dtype=jnp.float32))]
        for j in range(W - 1):
            w0.append(p0_input(jnp.zeros((Km, num_label), dtype=jnp.float32)))
            w1.append(p1_input(jnp.ones((Km, num_label), dtype=jnp.float32)))
        B0, B1 = [p0_input(jnp.zeros((H, Kj), dtype=jnp.int32))], [p1_input(jnp.ones((H, Kj), dtype=jnp.int32))]
        for j in range(W - 1):
            B0.append(p0_input(jnp.zeros((H, Km), dtype=jnp.int32)))
            B1.append(p1_input(jnp.ones((H, Km), dtype=jnp.int32)))
        w, B = [], []
        for j in range(W):
            w.append(ppd.device("SPU")(lambda w0, w1: w0 + w1)(w0[j], w1[j]))
            B.append(ppd.device("SPU")(lambda B0, B1: B0 + B1)(B0[j], B1[j]))
        
        transformation_start = time.time()
        res = []
        for j in range(W):
            temp = ppd.device("SPU")(naive_transformation)(B[j], w[j])
            res.append(ppd.get(temp))
        print("- microbenchmark naive transformation time: {:.2f} s".format(time.time() - transformation_start))
    print("- total time: {:.2f} s".format(time.time() - start_time))
    
def exp_iv_gcd_vp():
    exp_H = 800
    exp_W = 20
    exp_K = 5
    
    print("-----------------------IV SETTING-----------------------")
    print(f"+ experiment parameters: H={exp_H}, W={exp_W}, K={exp_K}")
    print(f"dataset: GCD, mode: vp")
    start_time = time.time()
    
    print("-----------------------COMPUTE WOE VP-----------------------")
    exp_vp(exp_H, exp_W, exp_K, exp_times=1)
    woe_time = time.time()
    
    print("-----------------------COMPUTE IV VP-----------------------")
    # (H, W * K)
    Wu = exp_W * exp_K // 2
    Wv = exp_W * exp_K - Wu
    BU = random.randint(random.PRNGKey(42), (exp_H, Wu), minval=0, maxval=2)
    BV = random.randint(random.PRNGKey(42), (exp_H, Wv), minval=0, maxval=2)
    label = random.randint(random.PRNGKey(42), (exp_H, 1), minval=0, maxval=2)
    woe_value_v = random.uniform(random.PRNGKey(42), (Wu, 1), minval=-3, maxval=3)
    woe_value_shared = random.uniform(random.PRNGKey(42), (Wv, 1), minval=-3, maxval=3)
    
    BU = p0_input(BU)
    BV = p1_input(BV)
    label = p1_input(BV)
    woe_value_v = p1_input(woe_value_v)
    woe_value_shared = co_input(woe_value_shared)
    
    iv_v = ppd.device("P2")(compute_iv)(BV, label, woe_value_v)
    iv_s = ppd.device("SPU")(compute_iv)(BU, label, woe_value_v)
    iv_v = ppd.get(iv_v)
    iv_s = ppd.get(iv_s)
    iv_time = time.time()
    print("- total time: {:.2f} s".format(iv_time - start_time))
    print("- woe time: {:.2f} s".format(woe_time - start_time))
    print("- iv time: {:.2f} s".format(iv_time - woe_time))
    

def exp_iv_gcd_hp():
    H = 800
    W_cat = 13
    K_cat = 5
    W_num = 7
    K_num = 5
    beta = jnp.arange(1, K_num) * 0.05
    
    print("-----------------------IV SETTING-----------------------")
    print(f"+ experiment parameters: H={H}, W_cat={W_cat}, K_cat={K_cat}, W_num={W_num}, K_num={K_num}, beta={beta}")
    print(f"dataset: GCD, mode: hp")
    start_time = time.time()
    
    print("-----------------------COMPUTE WOE HP-----------------------")
    exp_hp_cat(H, W_cat, K_cat, exp_times=1)
    exp_hp_num(H, W_num, K_num, exp_ALPHA, beta, exp_times=1)
    woe_time = time.time()
    
    print("-----------------------COMPUTE IV HP-----------------------")
    # (H, W * K)
    B = random.randint(random.PRNGKey(42), (H, W_cat * K_cat + W_num * K_num), minval=0, maxval=2)
    label = random.randint(random.PRNGKey(42), (H, 1), minval=0, maxval=2)
    woe_value = random.uniform(random.PRNGKey(42), (W_cat * K_cat + W_num * K_num, 1), minval=-3, maxval=3)
    
    B = co_input(B)
    label = co_input(label)
    woe_value = co_input(woe_value)
    
    iv = ppd.device("SPU")(compute_iv)(B, label, woe_value)
    iv = ppd.get(iv)
    iv_time = time.time()
    print("- total time: {:.2f} s".format(iv_time - start_time))
    print("- woe time: {:.2f} s".format(woe_time - start_time))
    print("- iv time: {:.2f} s".format(iv_time - woe_time))
    
    

def exp_iv_hcdr_vp():
    exp_H = 307511
    exp_W_cat = 51
    exp_K_cat = 5
    exp_W_num = 69
    exp_K_num = 10
    exp_beta = jnp.arange(1, exp_K_num) * 0.05
    global exp_FLAG_LARGE_DATASET
    exp_FLAG_LARGE_DATASET = True
    
    print("-----------------------IV SETTING-----------------------")
    print(f"+ experiment parameters: H={exp_H}, W={exp_W}, K={exp_K}")
    print(f"dataset: HCDR, mode: vp")
    start_time = time.time()
    
    print("-----------------------COMPUTE WOE VP-----------------------")
    exp_vp(exp_H, exp_W_cat, exp_K_cat, exp_times=1)
    exp_vp(exp_H, exp_W_num, exp_K_num, exp_times=1)
    woe_time = time.time()
    
    print("-----------------------COMPUTE IV VP-----------------------")
    # (H, W * K)
    Wu = (exp_W_cat * exp_K_cat + exp_W_num * exp_K_num) // 2
    Wv = (exp_W_cat * exp_K_cat + exp_W_num * exp_K_num) - Wu
    BU = random.randint(random.PRNGKey(42), (exp_H, Wu), minval=0, maxval=2)
    BV = random.randint(random.PRNGKey(42), (exp_H, Wv), minval=0, maxval=2)
    label = random.randint(random.PRNGKey(42), (exp_H, 1), minval=0, maxval=2)
    woe_value_u = random.uniform(random.PRNGKey(42), (Wu, 1), minval=-3, maxval=3)
    woe_value_v = random.uniform(random.PRNGKey(42), (Wv, 1), minval=-3, maxval=3)
    
    BU = p0_input(BU)
    BV = p1_input(BV)
    label = p1_input(label)
    woe_value_u = co_input(woe_value_u)
    woe_value_v = p1_input(woe_value_v)
    
    iv_u = ppd.device("SPU")(compute_iv)(BU, label, woe_value_u)
    iv_v = ppd.device("SPU")(compute_iv)(BV, label, woe_value_v)
    iv_u = ppd.get(iv_u)
    iv_v = ppd.get(iv_v)
    iv_time = time.time()
    print("- total time: {:.2f} s".format(iv_time - start_time))
    print("- woe time: {:.2f} s".format(woe_time - start_time))
    print("- iv time: {:.2f} s".format(iv_time - woe_time))
    
    
def exp_iv_hcdr_hp():
    H = 307511
    W_cat = 51
    K_cat = 5
    W_num = 69
    K_num = 10
    beta = jnp.arange(1, K_num) * 0.05
    global exp_FLAG_LARGE_DATASET
    exp_FLAG_LARGE_DATASET = True
    
    print("-----------------------IV SETTING-----------------------")
    print(f"+ experiment parameters: H={H}, W_cat={W_cat}, K_cat={K_cat}, W_num={W_num}, K_num={K_num}, beta={beta}")
    print(f"dataset: HCDR, mode: hp")
    start_time = time.time()
    
    print("-----------------------COMPUTE WOE HP-----------------------")
    exp_hp_cat(H, W_cat, K_cat, exp_times=1)
    exp_hp_num(H, W_num, K_num, exp_ALPHA, beta, exp_times=1)
    woe_time = time.time()
    
    print("-----------------------COMPUTE IV HP-----------------------")
    # (H, W * K)
    B = random.randint(random.PRNGKey(42), (H, W_cat * K_cat + W_num * K_num), minval=0, maxval=2)
    label = random.randint(random.PRNGKey(42), (H, 1), minval=0, maxval=2)
    woe_value = random.uniform(random.PRNGKey(42), (W_cat * K_cat + W_num * K_num, 1), minval=-3, maxval=3)
    
    B = p0_input(B)
    label = p1_input(label)
    woe_value = co_input(woe_value)
    
    iv = ppd.device("SPU")(compute_iv)(B, label, woe_value)
    iv = ppd.get(iv)
    iv_time = time.time()
    print("- total time: {:.2f} s".format(iv_time - start_time))
    print("- woe time: {:.2f} s".format(woe_time - start_time))
    print("- iv time: {:.2f} s".format(iv_time - woe_time))
    

if __name__ == "__main__":
    exp_FLAG_LARGE_DATASET = exp_H * exp_W * exp_K > 100000 * 100 * 10
    if exp_MODE == "vp":
        exp_vp(exp_H, exp_W, exp_K, exp_times=exp_TIMES)
    elif exp_MODE == "hp_cat":
        exp_hp_cat(exp_H, exp_W, exp_K, exp_times=exp_TIMES)
    elif exp_MODE == "hp_num":
        exp_BETA = jnp.arange(1, exp_K) * 0.05
        exp_hp_num(exp_H, exp_W, exp_K, exp_ALPHA, exp_BETA, exp_times=exp_TIMES)
    elif exp_MODE == "mbm_appqua":
        exp_mbm_appqua(exp_H, exp_W, exp_K, exp_ALPHA, jnp.array(exp_BETA), exp_times=exp_TIMES)
    elif exp_MODE == "mbm_transformation":
        exp_mbm_transformation(5, exp_K, exp_H, exp_W, exp_times=exp_TIMES)
    elif exp_MODE == "mbm_naive_transformation":
        exp_mbm_naive_transformation(5, exp_K, exp_H, exp_W, exp_times=exp_TIMES)
    elif exp_MODE == "iv_gcd_vp":
        exp_iv_gcd_vp()
    elif exp_MODE == "iv_gcd_hp":
        exp_iv_gcd_hp()
    elif exp_MODE == "iv_hcdr_vp":
        exp_iv_hcdr_vp()
    elif exp_MODE == "iv_hcdr_hp":
        exp_iv_hcdr_hp()