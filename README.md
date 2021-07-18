# What's this?
Pytorch implementaion of "Minimum Sharpness"

proposed in [Minimum sharpness: Scale-invariant parameter-robustness of neural networks](https://arxiv.org/abs/2106.12612)

# Note
This is the simplified implementation to clarify paper contribution. If you want to know more about effcient computation of tr[H] and diag[H], please refer to this [well-maintained repository](https://gitlab.com/takuo-h/fast-exact-trh/) by [Takuo Hamaguchi](https://takuo-h.github.io).

# Results you will see
<img src="https://github.com/ibayashi-hikaru/minimum-sharpness/blob/main/00-check-effective-calculation/VIEW/proposal-accuracy.png" height="300">            <img src="https://github.com/ibayashi-hikaru/minimum-sharpness/blob/main/02-sharpness-comparison/VIEW/model%3DLeNet/proposal-sharpness.png" height="300">

# How to use
 

To run "Randomized label experiment" in the paper, execute the following commands

```bash
experiment/run.py
```

Next, to visualize the results, execute the following commands

```bash
visualize/run.py
```

