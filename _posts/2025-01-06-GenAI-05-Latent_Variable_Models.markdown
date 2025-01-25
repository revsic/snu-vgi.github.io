---
layout: post
title:  "GenAI Lecture: 05 Latent Variable Models"
date:   2025-01-05 01:09:12 +0900
categories: Gen AI Lecture
---

> 들어가기: 본 블로그 글은 S. Ermon, Y. Song, CS236 Deep Generative Models, Stanford University의 강의를 참고하여 제작되었습니다.

### Recap

지금까지 우리가 배운 모델들을 정리해보면 다음과 같다.

- Autoregressive models
    - 체인 룰을 이용한 factorization은 일반화가 가능하다
    - 변수간 조건부를 독립적으로 두거나 뉴럴 네트워크의 설계 의존을 바꿈으로 인해 간결한 표현이 가능하다
- Autoregressive models 장점
    - Likelihoods의 값을 쉽게 얻을 수 있다
    - 학습이 쉽다.
- Autoregressive models 단점
    - 변수간에 미리 순서의 정의가 필요하다
    - 생성이 병렬적이지 않고 연쇄적으로 일어나야 한다
    - 지도학습 없이는 의미 있는 특징 (feature)를 배울 수 없다.

### Latent Variable Models: Motivation

[얼굴 사진]

우리가 사람의 얼굴 사진을 보면 이미지 $$x$$로 부터 성별, 눈의 색깔, 머리카락 색생, 자세 등등 으로 부터 다양성을 확인할 수 있다. 하지만 사람이 이러한 특성을 일일이 마킹해 놓지 않으면, 직접적으로 활용할 수는 없다. 즉 잠재적 (latent)이다.

다른 그림을 살펴보자.

[다른 그림]

오직 위의 그림에서 처럼 색으로 표현된 부분만 데이터에서 확인이 가능하다. 이미지로 생각하면 픽셀의 값이다. 이미지에서 직접 볼 수 없는 잠재적인 변수 $$z$$는 높은 수준의 특징값을 가지고 있다. 사람의 얼굴로 생각해보면, 앞서 언급한 성별, 눈의 색깔, 머리 색상 등등이다.

따라서, 잠재 변수에 대해 다음과 같이 생각해볼 수 있다.
- 만약 $$z$$가 적절히 선택되었다면, $$p(\mathrm{x} \mid \mathrm{z})$$는 $$p(\mathrm{x})$$보다 훨씬 간결해야 한다.
- 우리가 이러한 모델을 학습했다고 하면, $$p(z \mid x)$$을 활용해 이러한 특성을 알아낼 수 있어야 한다. 예를 들면, $$p(EyeColor=Blue \mid x)$$ 처럼.

**Challenge**이러한 점을 사람이 일일이 표현하는 것은 매우 어렵다.


### Deep Latent Variable Models

뉴럴 네트워크를 활용하면, deep latent variable을 처리하는 모델이 된다. 
1. $$\mathrm{z} \sim \mathcal{N}(0, I)$$로 잠재 변수를 간단히 정의한다.
2. $$p(\mathrm{x} \mid \mathrm{z})=\mathcal{N}\left(\mu_{\theta}(\mathrm{z}), \Sigma_{\theta}(\mathrm{z})\right)$$ 는 $$\mu_{\theta}, \Sigma_{\theta}$$은 뉴럴 네트워크이다.

학습이 끝나고 나서 $$\mathrm{z}$$는 의미있는 잠재 요소와 대응되게 된다. 즉, 특징을 알아내게 되고, 궁극적으로 비지도 표현 학습 (unsupervised representation learning)이 된다. 이전과 같이 특징은 $$p(z \mid x)$$을 이용해 계산할 수 있다. 

### Mixture of Gaussians: a Shallow Latent Variable Model

가장 간단한 '얕은' 잠재 변수 모델로 가우시안 혼합 모델을 생각해보자. 우리는 다음과 같은 베이시안 네트워크를 만들 것이다. 

$$\mathrm{z} \rightarrow \mathrm{x}$$

여기서 잠재 변수 $$\mathrm{z}$$와 잠재 변수를 이용한 조건부 확률 $$p(\mathrm{x} \mid \mathrm{z}=k)$$는 다음과 같이 정의된다.

$$
\begin{align}
\mathrm{z} \sim Categorical (1, \cdots, K)\\
p(\mathrm{x} \mid \mathrm{z}=k) = \mathcal{N}\left(\mu_{k}, \Sigma_{k}\right)
\end{align}
$$

아주 간단한 카테고리 분포와 가우시안의 조합이지만, 우리가 방금 만든 것은 잠재적 변수를 활용한 생성 모델이다. 이 모델을 활용해서 어떻게 새로운 변수를 생성할 수 있을까?

**생성과정**
- $$z$$를 샘플링하여 가우시안 혼합 요소 $$k$$를 고른다.
- 선택된 가우시안으로 부터 데이터 포인트를 얻어낸다.

문제는 이렇게 만든 카테고리 분포 + 가우시안 분포 기반의 생성 모델이 얼마나 잘 데이터셋의 분포를 잘 따르는지 확인하는 길이다. 이것을 **클러스터링** 관점에서 보면 사후확률 (posterior) $$p(z \mid x)$$는 데이터 $$x$$에 대해 가우시안 혼합 변수를 의미한다. **비지도 학습** 관점에서 본다면, 레이블이 없는 데이터에서 무언가를 배우기를 희망한다. 하지만 이런 문제는 ill-posed problem이 되기 때문에 직접 해결은 어렵다.

[그림 삽입]

예를 들어 손으로 쓴 숫자 (hand-written digits)를 비지도 군집화를 하게 되면 다음과 같은 결과를 볼 수 있다. 

[그림 삽입]

또다른 관점에서 우리가 만들고 있는 모델들을 보면, 간단한 가우시안 분포 하나로 부터 좀 더 복잡한 분포를 만들고 있음을 알 수 있다.

$$
p(\mathrm{x})=\sum_{\mathrm{z}} p(\mathrm{x}, \mathrm{z})=\sum_{\mathrm{z}} p(\mathrm{z}) p(\mathrm{x} \mid \mathrm{z})=\sum_{k=1}^{K} p(\mathrm{z}=k) \underbrace{\mathcal{N}\left(\mathrm{x} ; \mu_{k}, \Sigma_{k}\right)}_{\text {component }}
$$

### Variational Autoencoder

무한개의 가우시안으로 표현을 해보면 어떨까?

- 이전과 마찬가지로, $$\mathrm{z} \sim \mathcal{N}(0, I)$$를 정의한다.
- $$p(x \mid z)=\mathcal{N}\left(\mu_{\theta}(z), \Sigma_{\theta}(z)\right)$$에서 $$\mu_{\theta}, \Sigma_{\theta}$$는 뉴럴 네트워크이다. 세부적으로는 다음과 같다.

$$
\begin{align}
\mu_{\theta}(\mathrm{z})= &\sigma(A z+c)=\left(\sigma\left(a_{1} z+c_{1}\right), \sigma\left(a_{2} z+c_{2}\right)\right)= \left(\mu_{1}(z), \mu_{2}(z)\right)\\
\Sigma_{\theta}(z) = &\operatorname{diag}(\exp(\sigma(Bz + d))) = \begin{pmatrix} \exp(\sigma(b_1z + d_1) & 0 \\ 0 & \exp(\sigma(b_2z + d_2))\end{pmatrix}\\
\theta=& (A, B, c, d)
\end{align}
$$

- 비록 $$p(\mathrm{x} \vert \mathrm{z})$$가 간단하지만, 주변 분포 $$p(\mathrm{x})$$는 매우 복잡하거나 유연한 분포를 표현할 수 있게 된다.


### Recap

정리해보면, 잠재적 변수 모델은 복잡한 분포 $$p(x)$$를 좀 더 간결한 빌딩 블록처럼 만들 수 있게 도와 준다. $$p(x \mid z)$$. 또한 비지도 학습과 관련이 되어 논리적으로 이치가 맞는다. (군집화, 비지도 표현 학습, 등)

다만, 모든게 손쉽게 해결 되지는 않는다. 완전히 관측 가능한 상태의 autoregressive model 대비 훨씬 학습이 어려워진다.

### Marginal Likelihood

[이미지 삽입]

만약 몇개의 픽셀 값이 학습시 보이지 않는다고 생각해보자. 위의 그림 참고. 그렇다면 $$X$$를 관측한 랜덤 변수라 정의하고, $$Z$$를 관측하지 않은 값 (숨겨진 변수 혹은 잠재적 변수)이라 칭하자. 그리고 우리가 다음과 같은 결합확률 분포를 만든다고 가정하자. 

$$
p(\mathrm{X}, \mathrm{Z} ; \theta)
$$

그렇다면 학습 데이터 $$\bar{x}$$로 부터 얻을 수 있는 확률 $$p(X=\bar{x} ; \theta)$$는 무엇이 될까?

$$
\sum_{\mathrm{z}} p(\mathrm{X}=\overline{\mathrm{x}}, \mathrm{Z}=\mathrm{z} ; \theta)=\sum_{\mathrm{z}} p(\overline{\mathrm{x}}, \mathrm{z} ; \theta)
$$

여기서 우리는 이미지를 완성하기 위해 가능한 모든 경우의 수를 생각해야 한다.

앞에서 활용하였던 가우시안을 활용해 보자.

- $$\mathrm{z} \sim \mathcal{N}(0, I)$$ 로 정의한다.
- $$p(x \mid z)=\mathcal{N}\left(\mu_{\theta}(\mathrm{z}), \Sigma_{\theta}(\mathrm{z})\right)$$ 로 정의하고 $$\mu_{\theta}, \Sigma_{\theta}$$를 뉴럴 네트워크로 만들자.
- $$\mathrm{Z}$$는 학습시 보이지 않는 변수이다.
- 우리가 결합 확률 분포를 생각한다면, 학습 데이터 $$\overline{\mathrm{X}}$$에 대해 $$p(\mathrm{X}=\overline{\mathrm{X}} ; \theta)$$는 무엇이 될까?

$$
\int_{\mathrm{z}} p(\mathrm{X}=\overline{\mathrm{x}}, \mathrm{Z}=\mathrm{z} ; \theta) d \mathrm{z}=\int_{\mathrm{z}} p(\overline{\mathrm{x}}, \mathrm{z} ; \theta) d \mathrm{z}
$$

### 일부만 관측되는 데이터에 대해

우리의 결합 확률 분포가 다음과 같다고 가정해보자.

$$
p(\mathrm{X}, \mathrm{Z} ; \theta)
$$

그러면 우리는 데이터 셋 $$\mathcal{D}$$이 있고, 각 데이터 샘플 $$X$$는 관측이 가능하다. 그리고 변수 $$Z$$는 관측이 불가능하다 (군집 혹은 군집의 번호 등) $$\mathcal{D}=\left\{\mathrm{x}^{(1)}, \cdots, \mathrm{x}^{(M)}\right\}$$

여기서 maximum likelihood learning을 하면 다음과 같다.

$$
\log \prod_{\mathrm{x} \in \mathcal{D}} p(\mathrm{x} ; \theta)=\sum_{\mathrm{x} \in \mathcal{D}} \log p(\mathrm{x} ; \theta)=\sum_{\mathrm{x} \in \mathcal{D}} \log \sum_{\mathrm{z}} p(\mathrm{x}, \mathrm{z} ; \theta)
$$

**문제점** 여기서 값을 $$\log \sum_{z} p(x, z ; \theta)$$ 직접 계산하는 것은 불가능하다. 왜냐하면 이진 이미지라 할 지라도 $$\sum_{z} p(x, z ; \theta)$$ 를 계산하기 위해서는 변수가 가질 수 있는 모든 경우의 수를 고려하여야 하고, 이것으로 인해, $$2^{30}$$번 만큼의 합을 해야 하기 때문이다.
랜덤 변수가 연속 변수일지라도 마찬가지이다. $$\log \int_{z} p(x, z ; \theta) d z$$를 직접 게산하는것은 불가능하다. 그리고 이런 경우 그래디언트 $$\nabla_{\theta}$$를 계산하는 것도 매우 어렵다.

그렇다면 어떤 방법을 쓰는 것이 좋을까? **추정**이 필요하다. 만일 데이터 샘플 하나 $$x \in \mathcal{D}$$로 부터 그래디언트를 계산한다면 어떨까? 물론 추정 기법은 매우 가볍고 계산하기 편해야 할 것이다.

### 첫번째 시도: 순진한 몬테카를로

우리는 likelihood 함수 $$p_{\theta}(\mathbf{x})$$가 일부만 보여진 데이터에 대해 계산하기 어렵다는 것을 잘 알고 있다.

$$
p_{\theta}(\mathbf{x})=\sum_{\text {All values of } \mathbf{z}} p_{\theta}(\mathbf{x}, \mathbf{z})=|\mathcal{Z}| \sum_{\mathbf{z} \in \mathcal{Z}} \frac{1}{|\mathcal{Z}|} p_{\theta}(\mathbf{x}, \mathbf{z})=|\mathcal{Z}| \mathbb{E}_{\mathbf{z} \sim \operatorname{Uniform}(\mathcal{Z})}\left[p_{\theta}(\mathbf{x}, \mathbf{z})\right]
$$

그렇다면, 사용이 가능할만한 기대값을 생각할 수 있다. 몬테카를로 기법을 써보자.

- 랜덤으로 $$\mathbf{z}^{(1)}, \cdots, \mathbf{z}^{(k)}$$ 변수를 추출한다.
- 샘플의 평균으로 기대값을 **추정**한다.

$$
\sum_{\mathbf{z}} p_{\theta}(\mathbf{x}, \mathbf{z}) \approx|\mathcal{Z}| \frac{1}{k} \sum_{j=1}^{k} p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(j)}\right)
$$

위의 방법은 이론적으로는 작동하지만, 실제로는 그렇지 않다. 대부분의 $$z$$에 대해서 $$p_{\theta}(\mathbf{x}, z)$$가 극도로 낮기 때문이다. (대부분의 이미지를 채워 넣은 경우가 실제 이미지와는 거리가 멀 것이다.) 몇개는 값이 클 수 있지만, 절대로 정확한 완성을 균등 분포함수로는 맞출 수 없을 것이기 때문이다. 

따라서 $$\mathbf{z}^{(j)}$$를 선택하기 위한 좀 더 현명한 방법이 필요하다. 

### 두번째 시도: 중요도 샘플링

우리는 일부만 보여진 데이터의 likelihood $$p_{\theta}(\mathbf{x})$$를 계산하는게 어렵다는 것을 잘 알고 잇다.

$$
p_{\theta}(\mathbf{x})=\sum_{\text {All possible values of } \mathbf{z}} p_{\theta}(\mathbf{x}, \mathbf{z})=\sum_{\mathbf{z} \in \mathcal{Z}} \frac{q(\mathbf{z})}{q(\mathbf{z})} p_{\theta}(\mathbf{x}, \mathbf{z})=\mathbb{E}_{\mathbf{z} \sim q(\mathrm{z})}\left[\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right]
$$

몬테카를로를 다시 활용해보자.

- 이번에는 샘플 $$\mathbf{z}^{(1)}, \cdots, \mathbf{z}^{(k)}$$를 $$q(\mathbf{z})$$에서 얻는다. (중요도 샘플링이다.)
- 샘플의 평균을 이용해 기대값을 계산한다.

$$
p_{\theta}(\mathbf{x}) \approx \frac{1}{k} \sum_{j=1}^{k} \frac{p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(j)}\right)}{q\left(\mathbf{z}^{(j)}\right)}
$$

그렇다면 $$q(\mathrm{z})$$를 위한 좋은 선택은 무엇일까. 직관적으로는 그럴듯한 완성을 하면 될 것이다. 그렇다면 log-likelihood를 다음과 같이 계산한다.

$$
\log \left(p_{\theta}(\mathbf{x})\right) \approx \log \left(\frac{1}{k} \sum_{j=1}^{k} \frac{p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(j)}\right)}{q\left(\mathbf{z}^{(j)}\right)}\right){\approx} \log \left(\frac{p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(1)}\right)}{q\left(\mathbf{z}^{(1)}\right)}\right)
$$

하지만, 아래와 같음이 자명하다.

$$
\mathbb{E}_{\mathbf{z}^{(1)} \sim q(\mathbf{z})}\left[\log \left(\frac{p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(1)}\right)}{q\left(\mathbf{z}^{(1)}\right)}\right)\right] \neq \log \left(\mathbb{E}_{\mathbf{z}^{(1)} \sim q(\mathbf{z})}\left[\frac{p_{\theta}\left(\mathbf{x}, \mathbf{z}^{(1)}\right)}{q\left(\mathbf{z}^{(1)}\right)}\right]\right)
$$

$$
\log \left(\sum_{\mathbf{z} \in \mathcal{Z}} p_{\theta}(\mathbf{x}, \mathbf{z})\right)=\log \left(\sum_{\mathbf{z} \in \mathcal{Z}} \frac{q(\mathbf{z})}{q(\mathbf{z})} p_{\theta}(\mathbf{x}, \mathbf{z})\right)=\log \left(\mathbb{E}_{\mathbf{z} \sim q(\mathrm{z})}\left[\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right]\right)
$$

로그는 볼록함수이다. 

$$
\log \left(t x+(1-t) x^{\prime}\right) \geq t \log (x)+(1-t) \log \left(x^{\prime}\right)
$$

여기서 젠슨 부등식을 활용하면 다음을 알 수 있다.

$$
\log \left(\mathbb{E}_{\mathrm{z} \sim q(\mathrm{z})}[f(\mathrm{z})]\right)=\log \left(\sum_{\mathrm{z}} q(\mathrm{z}) f(\mathrm{z})\right) \geq \sum_{\mathrm{z}} q(\mathrm{z}) \log f(\mathrm{z})
$$

그렇다면 지금까지 유도한 수식을 정리해보자. 

$$
f(z)=\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q(z)}
$$

를 사용하면 다음과 같아진다. 

$$
\log \left(\mathbb{E}_{\mathbf{z} \sim q(\mathrm{z})}\left[\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q(\mathrm{z})}\right]\right) \geq \mathbb{E}_{\mathbf{z} \sim q(\mathrm{z})}\left[\log \left(\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right)\right]
$$

이것은 **Evidence Lower Bound (ELBO)**라고 부른다.

### Variational Inference

만약 $$q(z)$$를 잠재변수 표현을 위한 임의의 확률 분포라고 가정하자.

**Evidence Lower Bound (ELBO)**는 어떠한 $$q$$에 대해서도 성립한다.

$$
\begin{align}

\log p(x;\theta) \geq & \sum_z q(z) \log\left(\frac{p_{\theta}(x, z)}{q(z)}\right)\\

= & \sum_{z} q(z) \log p_\theta (x, z) ~\underbrace{- \sum_z q(z) \log q(z)}_{\mathrm{Entropy}~H(q)~\mathrm{of}~q}\\

= & \sum_z q(z)\log p_\theta (x, z) + H(q)

\end{align}
$$

여기서 $$q=p(z \vert x ; \theta)$$라면 다음이 성립한다.

$$
\log p(\mathrm{x} ; \theta)=\sum_{\mathrm{z}} q(\mathrm{z}) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H(q)
$$

(첨언) 여기서 얻는 값은 우리가 EM 알고리즘에서 얻는 E 스텝과 같다. 

이러한 bound는 tight하다. $$q(\mathrm{z})=p(\mathrm{z} \mid \mathrm{x} ; \theta)$$라면 바운드는 다음과 같다.

$$
\begin{align}

\sum_z p(z | x; \theta) \log \frac{p(x, z; \theta)}{p(z | x ; \theta)} = & \sum p(z | x; \theta) \log \frac{p(z | x; \theta)p(x; \theta)}{p(z | x; \theta)}\\

= & \sum_{{z}} p({z} | {x} ; \theta) \log p({x} ; \theta)\\

= & \log p({x} ; \theta) \underbrace{\sum_{{z}} p({z} | {x} ; \theta)}_{=1} = \log p({x} ; \theta)

\end{align}
$$

위의 수식은 이전의 가중치 반영 샘플링에 한가지 중요한 사실을 알려준다. **우리는 반드시 그럴싸 한 이미지 완성**을 선택해야 한다. 

만약에 $$p(z | x ; \theta)$$가 계산하기 어렵다면 어떻게 해야 할까? 바운드가 얼마나 간격이 있을까?

앞에서 $$q(z)$$가 어떠한 확률 분포도 가능하다고 정의했다. 약간의 선형대수를 사용하면 다음과 같다.

$$
D_{K L}(q(\mathrm{z}) \| p(\mathrm{z} | \mathrm{x} ; \theta)) = -\sum_{\mathrm{z}} q(\mathrm{z}) \log p(\mathrm{z}, \mathrm{x} ; \theta)+\log p(\mathrm{x} ; \theta)-H(q) \geq 0
$$

식을 조정해서 다름과 같은 ELBO를 얻는다.

$$
\log p(\mathrm{x} ; \theta) \geq \sum_{\mathrm{z}} q(\mathrm{z}) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H(q)
$$

여기서 $$q=p(\mathrm{z} \mid \mathrm{x} ; \theta)$$로 두면 $$D_{K L}\left(q(\mathrm{z}) \parallel p(\mathrm{z} | \mathrm{x} ; \theta)\right)=0$$이기 때문에 다음이 만족한다.

$$
\log p(\mathrm{x} ; \theta)=\sum_{\mathrm{z}} q(\mathrm{z}) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H(q)
$$

일반적으로 다음을 만족한다.

$$\log p(\mathrm{x} ; \theta)=\mathrm{ELBO}+D_{K L}\left(q(\mathrm{z}) \parallel p(\mathrm{z} | \mathrm{x} ; \theta)\right)$$

즉 $$q(z)$$가 $$p(z \mid x ; \theta)$$에 가까울수록, ELBO는 정확한 log-likelihood에 점점 더 가까워진다.

만약 사후확률 $$p(z | x ; \theta)$$이 계산하기 어렵다면 어떻게 해야 할까?
$$q(\mathrm{z} ; \phi)$$가 계산이 유용한 확률 분포라고 가정해보자. 이 분포는 $$\phi$$로 표현이 되고, variational parameter로 표현한다.

예를 들어 가우시안일 경우, 평균과 분산 행렬을 $$\phi$$를 통해 나타낸다.

$$
q(\mathrm{z} ; \phi)=\mathcal{N}\left(\phi_{1}, \phi_{2}\right)
$$

**variational inference**: $$q(z ; \phi)$$가 최대한 $$p(z | x ; \theta)$$가 가까워지는 $$\phi$$를 고른다. 

[그림 삽입]

그렇다면, 사후확률 $$p(z | x ; \theta)$$이 $$\mathcal{N}(2,2)$$ (오렌지색 확률 분포) 보다 $$\mathcal{N}(-4,0.75)$$ (초록색 확률 분포)로 좀 더 표현이 잘 됨을 알 수 있다.

