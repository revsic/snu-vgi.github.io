---
layout: post
title:  "GenAI Lecture: 02 Representation"
date:   2025-01-05 01:09:12 +0900
categories: Gen AI Lecture
---

> 들어가기: 본 블로그 글은 S. Ermon, Y. Song, CS236 Deep Generative Models, Stanford University의 강의를 참고하여 제작되었습니다.

### Continuous Random Variables

지금까지는 주로 discrete한 random variable에 대해 이야기를 하였다 (동전의 앞 뒷면, 주사위의 숫자, ). Continuous randomvariable을 확률 분포로 나타낸다면 고려해야 할 점은 없을까?
우선은 probablity density function이 필요하다. 이제는 베이시안 네트워크의 테이블 구조로 나타낼 수는 없고, 몇가지 파라미터로 명확하게 표현 가능한 density function으로 표현을 해야 한다. 

예를 들면 다음과 같다.
- **가우시안 분포**: $$X \sim \mathcal{N}(\mu, \sigma)$$ 일때 $$p_{X}(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-(x-\mu)^{2} / 2 \sigma^{2}}$$
- **균일 분포**: $$X \sim \mathcal{U}(a, b)$$ 일때 $$p_{X}(x)=\frac{1}{b-a} 1[a \leq x \leq b]$$
- 기타 다른 연속 분포

만약 $$X$$가 연속적인 랜덤 벡터일 경우에는 (dimension이 1이상인 경우), 결합확률 분포를 이용해 쉽게 표현할 수 있다. 

$$p_{X}(\mathbf{x})=\frac{1}{\sqrt{(2 \pi)^{n}|\Sigma|}} \exp \left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{T} \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

랜덤 변수가 연속적이더라도, 랜덤 벡터일 경우에도 여전히 체인 룰, 베이스 룰 등은 그대로 적용이 가능하다. 예를 들면 다음과 같다.

$$
p_{X, Y, Z}(\mathbf{x}, \mathbf{y}, \mathbf{z})=p_{X}(\mathbf{x}) p_{Y \mid X}(\mathbf{y} \mid \mathbf{x}) p_{Z \mid\{X, Y\}}(\mathbf{z} \mid \mathbf{x}, \mathbf{y})
$$

### Continuous and Discrete Random Variables

**Mixture of 2 Gaussians**: 물론 discrete 랜덤변수와 continuous 랜덤변수를 함께 사용하는 것도 가능하다. 예를 들면 베르누이 분포와 가우시안 분포를 결합하는 예제는 다음과 같다.

$$Z \rightarrow X$$

결합 확률 분포는 다음과 같이 정의 된다. 

$$p_{Z, X}(z, x)=p_{Z}(z) p_{X \mid Z}(x \mid z)$$

그리고 각 랜덤 변수 $$Z$$와 $$X$$는 다음과 같이 정의된다.

$$Z \sim \operatorname{Bernoulli}(p)$$

- $$X|(Z=0) \sim \mathcal{N}\left(\mu_{0}, \sigma_{0}\right)$$
- $$X|(Z=1) \sim \mathcal{N}\left(\mu_{1}, \sigma_{1}\right)$$

이러한 결합분포 $$p_{Z, X}(z, x)$$ 표현하기 위해서는 다음과 같은 파라미터들이 필요함을 알 수 있다.

$$p, \mu_{0}, \sigma_{0}, \mu_{1}, \sigma_{1}$$

**Mixture of continuous random variables**: 다음 예시로서 두개의 연속 분포도 결합이 가능하다. 예를 들어, 균등 분포와 가우시안 분포는 다음과 같이 결합 가능하다. 

$$Z \rightarrow X$$

마찬가지로 결합 확률 분포는 다음과 같이 정의된다.

$$p_{Z, X}(z, x)=p_{Z}(z) p_{X \mid Z}(x \mid z)$$

각 랜덤 변수는 다음과 같이 정의된다.
- $$Z \sim \mathcal{U}(a, b)$$
- $$X \mid(Z=z) \sim \mathcal{N}(z, \sigma)$$

여기서 필요한 파라미터는 $$a, b, \sigma$$이다.

**Variational autoencoders**: 한가지 응용 사례로 VAE가 있다. 

$$Z \rightarrow X$$

결합확률 분포는 다음과 같다.

$$p_{Z, X}(z, x)=p_{Z}(z) p_{X \mid Z}(x \mid z)$$

각 랜덤 변수는 다음과 같이 정의된다.
- $$Z \sim \mathcal{N}(0,1)$$
- $$X \mid(Z=z) \sim \mathcal{N}\left(\mu_{\theta}(z), e^{\sigma_{\phi}(z)}\right)$$

여기서 $$\mu_{\theta}: \mathbb{R} \rightarrow \mathbb{R}$$과 $$\sigma_{\phi}$$는 둘 다 뉴럴 네트워크이다. 각각의 네트워크는 파라미터 $$\theta, \phi$$로 컨트롤 된다.

재미있는 사실은 $$\mu_{\theta}, \sigma_{\phi}$$이 매우 깊은 뉴럴 네트워크 이더라도 함수적으로는 여전히 가우시안 분포라는 점이다.