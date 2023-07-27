# Diffusion Model
It consist of two process:
1. Forward Process
2. Reverse Process

**Forward Process** - In the forward process we add noise to the input image. We don't add same amount of noise each time, 
this regulated by the scheduler which scales mean and var of noise. This ensures variance does not explode as we add more and more noise.
Different Scheduler:
1. Linear Scheduler
2. Cosine Scheduler

**Reverse Process** - 

**Model:**
```
DDPM
```

- [U-Net](https://arxiv.org/pdf/1505.04597.pdf) like architecture, it takes image ans input and using ResNet block and down-sample block project image to small resolution.
- After bottleneck it uses up-sample project back to original image size.
- Author puts attention block at bottlenek and also included skip connection between layers of same resolution.
- Sinusoidal time embedding is projected into each residual block for forward and reverse diffusion process for making at different time steps.

```
Diffusion Models Beat GANs on Image Synthesis
```
*Updates*
- Increasing depth versus width, holding model size relatively constant.
- Increasing the number of attention heads.
- Using attention at 32×32, 16×16, and 8×8 resolutions rather than only at 16×16.
- Using the BigGAN [5] residual block for upsampling and downsampling the activations.
- Rescaling residual connections with $1/\sqrt{2}$.
- Classifier Guidance
- Adaptive Group Normalization

  ![Adaptive Group Normalization]()

  $y_{s}$ : linear projection of time step; 
  $y_{b}$ : linear projection of class label
   
Model Prediction - Noise of Image

**Why Reverse Process of Diffusion Model removes noise step by step?**

Learning in this framework involves estimating small perturbations to a diffusion process.
Estimating small perturbations is more tractable than explicitly describing the full
distribution with a single, non-analytically-normalizable,
potential function. Furthermore, since a diffusion process
exists for any smooth target distribution, this method can
capture data distributions of arbitrary form. [Deep Unsupervised Learning]

# Math behind Diffusion Models
## Notation :
- $x_{t}$ : Image after t iteration of adding noise.
- $x_{T}$ : Final image following isotropic Gaussian
- $q(x_{t}| x_{t-1})$ : Forward Process
- $p(x_{t-1}| x_{t})$ : Reverse Process 
- $β_{t}$ : Variance schedule, range[0,1], start small and gets increased
- $α_{t}$ =  $1 - β_{t}$
- $\bar{α_{t}}$ = $\prod_{s=1}^{t} α_{s}$ 
- $ϵ_{t} \sim N(0, I)$
- $bar{ϵ}_{t-2}$ : merage of two gaussian 

## Equations : 

- $q(x_{t}| x_{t-1}) = \mathcal{N}(x_{t}; \sqrt{1-β_{t}} \cdot  x_{t-1}, β_{t}I)$

  Above equation adds noise step by step. There is a better way of doing it. 
  $q(x_{t}| x_{t-1}) = \mathcal{N}(x_{t}; \sqrt{1-β_{t}} \cdot  x_{t-1}, β_{t}I)$

  $= \sqrt{1-β_{t}} \cdot  x_{t-1}+ \sqrt{β_{t}} \cdot ϵ_{t-1}$  
  $= \sqrt{α_{t}} \cdot  x_{t-1}+ \sqrt{1-α_{t}} \cdot ϵ_{t-1}$
  $= \sqrt{ α_{t} \cdot α_{t-1}} \cdot  x_{t-2}+ \sqrt{1-α_{t} \cdot α_{t-1}} \cdot \bar{ϵ}_{t-2}$

  $= \cdot \cdot \cdot$

  $= \sqrt{ \bar{α_{t}} } \cdot  x_{0}+ \sqrt{1-\bar{α_{t}}} \cdot ϵ$

  which results in next equation
  
- $q(x_{t}| x_{0}) = \mathcal{N}(x_{t}; \sqrt{\bar{α_{t}}} \cdot  x_{0}, \bar{α_{t}}I)$
- $p(x_{t-1}| x_{t}) =  \mathcal{N}(x_{t}; \mu_{Θ}(x_{t}, t),\Sigma_{Θ}(x_{t}, t) )$ 

  In tis case we have two parameters which characterize the normal distribution.

  We fix the variance to certain schedule. We don't need to predict it.

# Loss
$Loss = -log(p_{\theta}(x_{0}))$ can't be used as it is intractable. Since $x-{0}$ is dependent on $x_{1}, x_{2},...x_{T}$.
So , we use the following loss function.
![VLB](.\colabimages\VLB.jpeg)
from first term to second term can be arrived by using Bayes Rule $P(A|B) = \frac{P(AB)}{P(B)}$. we know,
![forward_process](.\colabimages\forward_process.jpeg)
![reverse_process](.\colabimages\reverse_process.jpeg)

Proceeding for further simplification:
![VLB](.\colabimages\VLB2.jpeg).
- $2^{nd}$ eq to $3^{rd}$ eq arrived by separating out $-log p_{\theta}(x_{T})$ term.
- $3^{rd}$ eq to $4^{th}$ eq arrived by separating out $-log \frac{q(x_{1}|x_{0})}{p(x_{1}|x_{0})}$.
- In $4^{th}$ eq ${q(x_{t}|x_{t-1})} = \frac{q(x_{t-1}|x_{t})\cdot q(x_{t})}{q(x_{t-1})}$. Each term in the RHS have really high variance since we don't know what we have really started with. So, we also make conditioning it to the $x_{0}$ which drtaically reduces the variance. ths lead from the $4^{th}$ eq to $5^{th}$ eq.
- In $5^{th}$ eq we have splitted the $3_{rd}$ term of $4_{th}$ eq splitting of log term 
- $6^{th}$ term to $7^{th}$ term arrived by simplifying  

# Resources
- [Deep Unsupervised Learning](https://arxiv.org/pdf/1503.03585.pdf)
- [DDPM](https://arxiv.org/pdf/2006.11239.pdf)
- [Blog Post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [minDiffusion github](https://github.com/cloneofsimo/minDiffusion)
- [Outlier YouTube](https://www.youtube.com/watch?v=HoKDTa5jHvg&t=1453s&ab_channel=Outlier)
