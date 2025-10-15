# https://zhuanlan.zhihu.com/p/588714270

#!/bin/sh
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######

# tc qdisc show dev lo

## replace DEV=lo with your card (e.g., eth0)
DEV=lo 

case "${1:-}" in
    del)
	    tc qdisc del dev $DEV root
        ;;
    lan)
        tc qdisc del dev $DEV root
        tc qdisc add dev $DEV root handle 1: tbf rate 1gbit burst 100000 limit 10000
        tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 1msec
        ;;
    wan)
        tc qdisc del dev "$DEV" root
        tc qdisc add dev "$DEV" root handle 1: tbf rate 160mbit burst 100000 limit 10000
        tc qdisc add dev "$DEV" parent 1:1 handle 10: netem delay 50msec
        ;;
    lst)
        tc qdisc show dev "$DEV"
        ;;
    *)
        echo "Usage: $0 {del|lan|wan|lst}"
        exit 1
        ;;
esac