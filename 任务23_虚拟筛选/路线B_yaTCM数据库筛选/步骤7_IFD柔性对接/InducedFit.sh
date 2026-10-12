#!/bin/bash
#SBATCH -n 64                           # 分配40个核心
#SBATCH --gres=gpu:0                   # 分配8个GPU资源
#SBATCH --time=7-00:00:00              # 设置作业最长运行时间为7天
#SBATCH -p public                        # 指定队列为 public
#SBATCH -o output_%j.out              # 将标准输出重定向到文件
#SBATCH -e error_%j.err               # 将错误输出重定向到文件

"${SCHRODINGER}/ifd" -NGLIDECPU 64 -NPRIMECPU 64 InducedFit.inp -NOLOCAL -HOST localhost -SUBHOST localhost -TMPLAUNCHDIR -WAIT