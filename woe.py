import argparse
import json

import jax
import jax.numpy as jnp
from jax import random
import spu
import time
import argparse

# exp_H           = [1000, 10000, 100000]    # 1K, 10K, 100K
# exp_W           = [10, 50, 100]
# exp_K           = [5, 10, 20]
exp_H           = [1000]    # 1K, 10K, 100K
exp_W           = [10]
exp_K           = [5]
exp_MIN_NUM_VAL = 1
exp_MAX_NUM_VAL = 4
exp_ALPHA       = 0.01
exp_BETA        = [0.5]
exp_OVERFLOW_FACTOR = 5

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
                
                raise Exception("Overflowed positive values")
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
                raise Exception("Overflowed positive values")
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
        self.K = jnp.max(max_logical_index - min_logical_index + 1).astype(jnp.int32)
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
        
        max_cats = jnp.max(X, axis=0)
        binary_features = []
        
        for feature_idx in range(W):
            n_classes = max_cats[feature_idx] + 1
            
            for class_val in range(n_classes):
                binary_feature = (X[:, feature_idx] == class_val).astype(jnp.int32)
                binary_features.append(binary_feature)
        
        return jnp.vstack(binary_features).T # (H, W = W * n_classes)
    
    def encode_hp_cat(self, X: jax.Array, max_cats: jax.Array):
        H, W = X.shape
        
        binary_features = []
        
        for feature_idx in range(W):
            n_classes = max_cats[feature_idx] + 1
            
            for class_val in range(n_classes):
                binary_feature = (X[:, feature_idx] == class_val).astype(jnp.int32)
                binary_features.append(binary_feature)
        
        return jnp.vstack(binary_features).T
    
    def encode_hp_num(self, I: jax.Array, U: jax.Array, V: jax.Array, alpha, beta, K, buckets_offset):
        quantile_counts = I.shape[1]
        
        # share U, V here
        X = jnp.vstack((U, V))  # (H, W)
        H, W = X.shape
        C = jnp.full((H, W, quantile_counts + 1), True) # (H, W, quantile_counts + 1)
        C = C.at[:, :, :-1].set(X[:, :, jnp.newaxis] < I[jnp.newaxis, :, :])
        binary_features = C[:, :, :-1] ^ C[:, :, 1:]  # (H, W, quantile_counts)
        
        return binary_features.squeeze(-1)
    

import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="conf/2pc.json")
args = parser.parse_args()

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
        zeta_pyu = ppd.device("P2")(woevp.get_zeta)(y, H)
        z_pos_bob_pyu, z_neg_bob_pyu = ppd.device("P2")(woevp.get_z)(BV, y)
        z_pos_bob_enc, z_neg_bob_enc, zeta_enc = z_pos_bob_pyu, z_neg_bob_pyu, zeta_pyu

        BU_pyu = BU
        y_pyu = ppd.device("P2")(lambda x: x)(y)
        z_pos_alice_enc, z_neg_alice_enc = ppd.device("SPU")(woevp.get_z)(BU_pyu, y_pyu)

        z_pos_enc, z_neg_enc = ppd.device("SPU")(lambda xp, xn, yp, yn: (jnp.vstack((xp, yp)), jnp.vstack((xn, yn))))(z_pos_alice_enc, z_neg_alice_enc, z_pos_bob_enc, z_neg_bob_enc)
        res_enc = ppd.device("SPU")(vectorized_sec_woe)(z_pos_enc, z_neg_enc, zeta_enc)
        res = ppd.get(res_enc)
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
        BU_pyu = ppd.device("P1")(data_encoder.encode_hp_cat)(U, max_cats)
        BV_pyu = ppd.device("P2")(data_encoder.encode_hp_cat)(V, max_cats)
        yu_pyu = ppd.device("P1")(lambda x: x)(yu)
        yv_pyu = ppd.device("P2")(lambda x: x)(yv)

        woehp = WoeHp()
        zeta_spu = ppd.device("SPU")(woehp.get_zeta)(H1, H2, yu_pyu, yv_pyu)
        z_pos_spu, z_neg_spu = ppd.device("SPU")(woehp.get_z)(BU_pyu, BV_pyu, yu_pyu, yv_pyu)

        res_enc = ppd.device("SPU")(vectorized_sec_woe)(z_pos_spu, z_neg_spu, zeta_spu)
        res = ppd.get(res_enc)
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
    dds.bucket_offset = dds.get_bucket_offset(jnp.min(X, axis=0), jnp.max(X, axis=0))
    dds.K = K
    
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
        z_pos_spu, z_neg_spu, zeta_spu = ppd.device("SPU")(woehp.get_all_for_num)(B, yu, yv)

        res_enc = ppd.device("SPU")(vectorized_sec_woe)(z_pos_spu, z_neg_spu, zeta_spu)
        res = ppd.get(res_enc)
        print("- woe hp num time: {:.2f} s".format(time.time() - start_time))
    print("- total time: {:.2f} s".format(time.time() - total_start_time))
    
# print("-----------------------VP setting-----------------------")
# print("+ split data")
# U, V, y = dataset_generator.split_vp(X, y, 0.5)
# W1, W2 = U.shape[1], V.shape[1]

# print("+ encode vp")
# data_encoder = DataEncoder()
# BU = ppd.device("P1")(data_encoder.encode_vp)(U)
# BV = ppd.device("P2")(data_encoder.encode_vp)(V)

# print("+ compute woe")
# woevp = WoeVp()
# zeta_pyu = ppd.device("P2")(woevp.get_zeta)(y, H)
# z_pos_bob_pyu, z_neg_bob_pyu = ppd.device("P2")(woevp.get_z)(BV, y)
# z_pos_bob_enc, z_neg_bob_enc, zeta_enc = ppd.device("SPU")(lambda x, y, z: (x, y, z))(z_pos_bob_pyu, z_neg_bob_pyu, zeta_pyu)

# BU_pyu = BU
# y_pyu = ppd.device("P2")(lambda x: x)(y)
# z_pos_alice_enc, z_neg_alice_enc = ppd.device("SPU")(woevp.get_z)(BU_pyu, y_pyu)

# z_pos_enc, z_neg_enc = ppd.device("SPU")(lambda xp, xn, yp, yn: (jnp.vstack((xp, yp)), jnp.vstack((xn, yn))))(z_pos_alice_enc, z_neg_alice_enc, z_pos_bob_enc, z_neg_bob_enc)
# res_enc = ppd.device("SPU")(vectorized_sec_woe)(z_pos_enc, z_neg_enc, zeta_enc)
# res = ppd.get(res_enc)


# print("-----------------------HP setting, cat-----------------------")
# print("+ split data")
# U, V, yu, yv = dataset_generator.split_hp(X, y, 0.5)
# H1, H2 = U.shape[0], V.shape[0]

# print("+ encode hp")
# data_encoder = DataEncoder()
# max_cats = jnp.max(X, axis=0) + 1
# BU_pyu = ppd.device("P1")(data_encoder.encode_hp_cat)(U, max_cats)
# BV_pyu = ppd.device("P2")(data_encoder.encode_hp_cat)(V, max_cats)
# yu_pyu = ppd.device("P1")(lambda x: x)(yu)
# yv_pyu = ppd.device("P2")(lambda x: x)(yv)

# print("+ compute z and zeta")
# woehp = WoeHp()
# zeta_spu = ppd.device("SPU")(woehp.get_zeta)(H1, H2, yu_pyu, yv_pyu)
# z_pos_spu, z_neg_spu = ppd.device("SPU")(woehp.get_z)(BU_pyu, BV_pyu, yu_pyu, yv_pyu)

# print("+ compute woe")
# res_enc = ppd.device("SPU")(vectorized_sec_woe)(z_pos_spu, z_neg_spu, zeta_spu)
# res = ppd.get(res_enc)


# print("-----------------------HP setting, num-----------------------")
# print("+ prepare params")
# alpha = exp_ALPHA
# beta = jnp.array(exp_BETA)
# beta = jnp.tile(beta, (W, 1))
# dds = DDSketch(alpha, beta)
# bucket_offset = dds.get_bucket_offset(jnp.min(X_num, axis=0), jnp.max(X_num, axis=0))
# dds.bucket_offset = bucket_offset
# K = exp_K
# dds.K = K

# print("+ split data")
# U, V, yu, yv = dataset_generator.split_hp(X_num, y_num, 0.5)
# H1, H2 = U.shape[0], V.shape[0]

# U = ppd.device("P1")(lambda x: x)(U)
# V = ppd.device("P2")(lambda x: x)(V)
# yu = ppd.device("P1")(lambda x: x)(yu)
# yv = ppd.device("P2")(lambda x: x)(yv)

# print("+ encode hp")
# data_encoder = DataEncoder()
# Su = ppd.device("P1")(dds.ddsketch_local)(U)
# Sv = ppd.device("P2")(dds.ddsketch_local)(V)
# I = ppd.device("SPU")(lambda Su, Sv: dds.ddsketch_global(Su, Sv, W))(Su, Sv)
# B = ppd.device("SPU")(lambda I, U, V: data_encoder.encode_hp_num(I, U, V, alpha, beta, K, bucket_offset))(I, U, V)

# print("+ compute z and zeta")
# woehp = WoeHp()
# z_pos_spu, z_neg_spu, zeta_spu = ppd.device("SPU")(woehp.get_all_for_num)(B, yu, yv)

# print("+ compute woe")
# res_enc = ppd.device("SPU")(vectorized_sec_woe)(z_pos_spu, z_neg_spu, zeta_spu)
# res = ppd.get(res_enc)

if __name__ == "__main__":
    for H in exp_H:
        for W in exp_W:
            for K in exp_K:
                exp_vp(H, W, K)
                exp_hp_cat(H, W, K)
                exp_hp_num(H, W, K, exp_ALPHA, jnp.array(exp_BETA))