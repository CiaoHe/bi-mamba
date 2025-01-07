[TOC]

# Implement detail

* Q: Implementation of Discretization B [issue 114](https://github.com/state-spaces/mamba/issues/114) [issue10](https://github.com/alxndrTL/mamba.py/issues/10)

  * 代码中没有用$\bar{B_t} = (∆A)^{-1}(exp(∆A)−I)\cdot ∆B$ 而是用了简化版的 $∆B$ (没有用原文中扯到的ZOH)

    > It makes the implementation slightly simpler without affecting empirical performance. One can think of it as a mix of ZOH for A and Euler discretization for B. 

* Throughput related
  * cg=True matters [issue90](https://github.com/state-spaces/mamba/issues/90), to reduce CPU/IO

* Parameters

  ![image-20240222171235795](/Users/kakusou/PaperComments/Images/image-20240222171235795.png)

  * 一个mamba block的参数量 (D: model_size, N: dstate, e: expand_factor, K: conv1d_kernel_size)

    * $2eD^2 + eD(K+1) + eD(D/16+2N) + eD(D/16+1) +  eDN + eD + eD^2$
    * 取e=2,  总参数量：$6.25D^2 + 5DN + 4D + 2D(K+1)$
    * 取N=16, K=4, 总参数量: $6.25D^2+94D$
    * In detail
  
      * Conv block: 
  
        $\#𝑝𝑎𝑟𝑎𝑚𝑠=(𝑘𝑒𝑟𝑛𝑒𝑙\_𝑠𝑖𝑧𝑒 \times |\frac{in\_channel}{groups}| + 1)×|𝑜𝑢𝑡\_𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠|$
  
  * 双向mamba
  
    ![image-20240222171305761](/Users/kakusou/PaperComments/Images/image-20240222171305761.png)
  
    * $6D^2 + 2(2D(K+1) + 0.25D^2 + 5DN + 4D)$
    * 取N=16, K=4, 总参数量: $6.5D^2+188D$
  
  * For reference: each transformer layer: $12D^2$

# Train **detail**

1. language modeling - scaling law (补充信息来自[issue144](https://github.com/state-spaces/mamba/issues/144))

   1. ![image-20240221214830835](/Users/kakusou/PaperComments/Images/image-20240221214830835.png)注意这里的lr是base，实际会x5

      * 2.8B model lr = 3xGPT3(1.6x1e-4) = 4.8x1e-4

   2. 原文Train recipes中写道

      > gradient clip = 1.0; 
      > weight decay = 0.1; 
      > no dropout; 
      > linear lr warmup + cosine decay (decay to 1e-5, peak value set $5\times$ GPT3 value)
      >
      > * 参考GPT-3的scaling设定
      >   * <img src="/Users/kakusou/PaperComments/Images/image-20240221215230005.png" alt="image-20240221215230005" style="zoom:67%;" /> 
      >
      > no bias term:
      > RMSNorm (instead of LayerNorm)
      > AdamW $\beta=(0.9, 0.95)$



# Install Micromamba

```bash
# py39 torch211+cu118: /cto_labas/AIDD/mamba/vllm.yaml

# 1. create new env [Name: prot]
1.9.0 prot python==3.9

# 2. install pytorch related
## pytorch version: 2.1.1+cu118
pip3 install torch==2.1.1+cu118 torchvision==0.16.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

## pytorch verison: 1.13.1+cu116
pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```
