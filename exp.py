import os
import re
import time
import signal
import argparse
import subprocess


PASS_EXP = False
if "http_proxy" in os.environ:
    del os.environ["http_proxy"]
if "https_proxy" in os.environ:
    del os.environ["https_proxy"]
    
    
NODECTL_QUIT_TIMEOUT = 20



"---------------------------------- Parser ----------------------------------"
parser = argparse.ArgumentParser(description="Quick experiment script.")
parser.add_argument("-s", "--test_suit", type=str, default="woe", help="which exp script to run")
parser.add_argument("-m", "--modes", type=str, nargs="+", help="subscript modes")
parser.add_argument("-t", "--times", type=int, default=1, help="number of experiments")
parser.add_argument("-H", default=[1000], nargs="+", type=int, help="number of samples")
parser.add_argument("-W", default=[50], nargs="+", type=int, help="num of features")
parser.add_argument("-K", default=[10], nargs="+", type=int, help="num of classes")
parser.add_argument("--alpha", default=0.01, type=float, help="error rate of DDSketch")
parser.add_argument("--beta", default=[0.5], nargs="+", type=float, help="list of partition ratio")

TEST_SUIT_MAP = {
    "woe": "woe.py",
}


"---------------------------------- Main ----------------------------------"
def quick_exp(test_suit: str, exp_modes: list, exp_times: int, exp_Hs: list, exp_Ws: list, exp_Ks: list, exp_alpha: float=0.01, exp_betas: list=[0.5]):
    '''
    Run the quick experiment.
    '''
    
    if not os.path.exists(f"output"):
        os.makedirs(f"output")
     
    config_path = "conf/2pc_alioth.json"
    exp_script = TEST_SUIT_MAP[test_suit]
                
    # console infomation.
    print(f"Here is the experiment setting:")
    print(f"Test suit:           {test_suit}")
    print(f"Script:              {exp_script}")
    print(f"Config file:         {config_path}")
    print(f"Modes:               {exp_modes}")
    print(f"Times:               {exp_times}")
    print(f"Hs:                  {exp_Hs}")
    print(f"Ws:                  {exp_Ws}")
    print(f"Ks:                  {exp_Ks}")
    print(f"Alpha:               {exp_alpha}")
    print(f"Betas:               {exp_betas}")
    print(f"------------------------------------------")
    
    # check if the config file exists.
    if not os.path.exists(config_path):
        raise Exception(f"Config file {config_path} not found.")
    
    for mode in exp_modes:
        for H in exp_Hs:
            for W in exp_Ws:
                for K in exp_Ks:
                    alpha = exp_alpha
                    beta = exp_betas
                    os.makedirs(f"output/{test_suit}/{mode}", exist_ok=True)
                    node_log_path = f"output/{test_suit}/{mode}/H{H}_W{W}_K{K}_node.log"
                    main_log_path = f"output/{test_suit}/{mode}/H{H}_W{W}_K{K}_main.log"
                    nodectl_cmd = f"python utils/nodectl.py --config {config_path} up"
                    task_cmd = f"python {exp_script} -m {mode} -H {H} -W {W} -K {K} -a {alpha} -b {' '.join([str(b) for b in beta])} -c {config_path} -t {exp_times}"
                    
                    print( "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print(f"Running experiment:  H={H}, W={W}, K={K}, alpha={alpha}, beta={beta}")
                    print(f"Mode:                {mode}")
                    print(f"nodectl cmd:         {nodectl_cmd}")
                    print(f"node log path:       {node_log_path}")
                    print(f"task cmd:            {task_cmd}")
                    print(f"main log path:       {main_log_path}")
                    
                    # open the subprecess.
                    with open(node_log_path, "w") as node_log_f, open(main_log_path, "w") as main_log_f:
                        node_log_f.write(""), main_log_f.write("")
                
                        nodectl = subprocess.Popen(
                            # nodectl_cmd + f" | tee {node_log_path}",  
                            nodectl_cmd.split(" "),  
                            stdout=node_log_f,
                            text=True)
                        time.sleep(2)
                        task = subprocess.Popen(
                            # task_cmd + f" | tee {main_log_path}",
                            task_cmd,
                            shell=True,
                            stdout=main_log_f,
                            text=True)
                
                        # wait util the process end.
                        start_time = time.time()
                        while True:
                            time.sleep(3)
                            print(f"Time:               {time.time() - start_time:<4.2f}s", end="\r")
                            if task.poll() is not None:
                                if nodectl.poll() is None:
                                    nodectl.send_signal(signal.SIGTERM)
                                else:
                                    print("\nDone.")
                                    break                
            

if __name__ == "__main__":
    args = parser.parse_args()
    test_suit = args.test_suit
    exp_modes = args.modes
    exp_times = args.times
    exp_Hs = args.H
    exp_Ws = args.W
    exp_Ks = args.K
    exp_alpha = args.alpha
    exp_betas = args.beta
    
    quick_exp(test_suit, exp_modes, exp_times, exp_Hs, exp_Ws, exp_Ks, exp_alpha, exp_betas)
    