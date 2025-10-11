# Copyright 2022 Ant Group Co., Ltd.
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

import argparse
import json
import time
import logging
import os
import signal
import sys

import spu.utils.distributed as ppd
from spu.utils.polyfill import Process

parser = argparse.ArgumentParser(description='SPU node service.')
parser.add_argument(
    "-c", "--config", default="examples/python/conf/3pc.json", help="the config"
)
subparsers = parser.add_subparsers(dest='command')
parser_start = subparsers.add_parser('start', help='to start a single node')
parser_start.add_argument("-n", "--node_id", default="node:0", help="the node id")
parser_up = subparsers.add_parser('up', help='to bring up all nodes')
parser_list = subparsers.add_parser('list', help='list node information')
parser_list.add_argument("-v", "--verbose", action='store_true', help="verbosely")


logger = logging.getLogger(__name__)

def _shutdown_workers(timeout=5):
    """Graceful shutdown of workers, with forced kill fallback."""
    global workers
    if not workers:
        return
    logger.info("Shutting down %d worker(s)...", len(workers))
    # request terminate
    for w in workers:
        try:
            if w.is_alive():
                logger.info("terminate worker pid=%s", getattr(w, "pid", "N/A"))
                w.terminate()
        except Exception as e:
            logger.exception("Error terminating worker: %s", e)

    # wait a bit
    end = time.time() + timeout
    for w in workers:
        remaining = max(0, end - time.time())
        try:
            w.join(timeout=remaining)
        except Exception:
            pass

    # force kill if still alive (best-effort)
    for w in workers:
        try:
            if w.is_alive():
                logger.warning("Worker still alive after terminate(): forcing kill pid=%s", getattr(w, "pid", "N/A"))
                try:
                    os.kill(w.pid, signal.SIGKILL)
                except Exception:
                    # Windows fallback: mp.Process.kill() available in py3.7+
                    try:
                        w.kill()
                    except Exception:
                        logger.exception("Failed to force-kill worker pid=%s", getattr(w, "pid", "N/A"))
        except Exception:
            pass

def _signal_handler(signum, frame):
    global terminate_requested
    logger.info("Received signal %s, shutting down...", signum)
    terminate_requested = True
    _shutdown_workers()
    # give a moment then exit
    sys.exit(0)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    nodes_def = conf["nodes"]
    devices_def = conf["devices"]
    
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.command == 'start':
        ppd.RPC.serve(args.node_id, nodes_def)
    elif args.command == 'up':
        workers = []
        
        try:
            for node_id in nodes_def.keys():
                worker = Process(target=ppd.RPC.serve, args=(node_id, nodes_def))
                worker.start()
                workers.append(worker)
                logger.info("Started worker for %s pid=%s", node_id, worker.pid)
                
            for w in workers:
                    # only block join for workers; if signal arrives, _signal_handler will exit
                    while w.is_alive():
                        try:
                            w.join(timeout=1.0)
                        except KeyboardInterrupt:
                            logger.info("KeyboardInterrupt caught, shutting down")
                            _shutdown_workers()
                            raise
        finally:
            _shutdown_workers()
            logger.info("All workers shutdown. Exiting.")

        for worker in workers:
            worker.join()
    else:
        parser.print_help()
