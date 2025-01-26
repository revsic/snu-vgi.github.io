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

만약에 $$p(z \vert x ; \theta)$$가 계산하기 어렵다면 어떻게 해야 할까? 바운드가 얼마나 간격이 있을까?

앞에서 $$q(z)$$가 어떠한 확률 분포도 가능하다고 정의했다. 약간의 선형대수를 사용하면 다음과 같다.

$$
D_{K L}(q(\mathrm{z}) \vert p(\mathrm{z} | \mathrm{x} ; \theta)) = -\sum_{\mathrm{z}} q(\mathrm{z}) \log p(\mathrm{z}, \mathrm{x} ; \theta)+\log p(\mathrm{x} ; \theta)-H(q) \geq 0
$$

식을 조정해서 다름과 같은 ELBO를 얻는다.

$$
\log p(\mathrm{x} ; \theta) \geq \sum_{\mathrm{z}} q(\mathrm{z}) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H(q)
$$

여기서 $$q=p(\mathrm{z} \mid \mathrm{x} ; \theta)$$로 두면 $$D_{K L}\left(q(\mathrm{z}) \parallel p(\mathrm{z} \vert \mathrm{x} ; \theta)\right)=0$$이기 때문에 다음을 만족한다.

$$
\log p(\mathrm{x} ; \theta)=\sum_{\mathrm{z}} q(\mathrm{z}) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H(q)
$$

**일반적으로 다음을 만족한다.**

$$\log p(\mathrm{x} ; \theta)=\mathrm{ELBO}+D_{K L}\left(q(\mathrm{z}) \parallel p(\mathrm{z} | \mathrm{x} ; \theta)\right)$$

즉 $$q(z)$$가 $$p(z \mid x ; \theta)$$에 가까울수록, ELBO는 정확한 log-likelihood에 점점 더 가까워진다.

만약 사후확률 $$p(z \vert x ; \theta)$$이 계산하기 어렵다면 어떻게 해야 할까?
$$q(\mathrm{z} ; \phi)$$가 계산이 유용한 확률 분포라고 가정해보자. 이 분포는 $$\phi$$로 표현이 되고, variational parameter로 표현한다.

예를 들어 가우시안일 경우, 평균과 분산 행렬을 $$\phi$$를 통해 나타낸다.

$$
q(\mathrm{z} ; \phi)=\mathcal{N}\left(\phi_{1}, \phi_{2}\right)
$$

**variational inference**: $$q(z ; \phi)$$가 최대한 $$p(z \vert x ; \theta)$$가 가까워지는 $$\phi$$를 고른다. 

[그림 삽입]

그렇다면, 사후확률 $$p(z \vert x ; \theta)$$이 $$\mathcal{N}(2,2)$$ (오렌지색 확률 분포) 보다 $$\mathcal{N}(-4,0.75)$$ (초록색 확률 분포)로 좀 더 표현이 잘 됨을 알 수 있다.

### 사후확률을 위한 variational approximation

[그림 삽입]

$$p\left(\mathrm{x}^{top}, \mathrm{x}^{bottom} ; \theta\right)$$ 확률을 숫자와 같은 이미지에게 높은 확률을 부여하는 분포라고 생각해보자. $$z=x^{top}$$는 보이지 않는 잠재적 변수이다.

만약 $$q\left(\mathrm{x}^{top} ; \phi\right)$$으로 정의하고 계산이 쉬운 확률 분포라고 가정해보면, $$x^{top}$$은 variational parameter $$\phi$$로 표현이 가능하다. 

$$
q\left(x^{top} ; \phi\right)= \prod_{\mathrm{unobserved~variable~x_i^{top}}}\left(\phi_{i}\right)^{x_{i}^{top}}\left(1-\phi_{i}\right)^{\left(1-x_{i}^{top}\right)}
$$

다음 세가지 경우에 대해 생각해보자. 

- $$\phi_{i}=0.5 \forall i$$이 좋은 $$p\left(x^{top} \mid x^{bottom} ; \theta\right)$$를 위한 좋은 추정일까?
- $$\phi_{i}=1.0 \forall i$$이 좋은 $$p\left(x^{top} \mid x^{bottom} ; \theta\right)$$를 위한 좋은 추정일까?
- $$\phi_{i} \approx 1$$이 숫자 9에 대한 좋은 추정값이라고 볼 수 있을까?

세번째 답이 가장 정확한 답이다.

### The Evidence Lower bound

앞서 언급한 ELBO에 대해 살펴보자.

[ELBO 그림 삽입]

$$
\begin{align}

\log p(x; \theta) \geq & \sum_z q(z; \phi) \log p(z, x;\theta) + H\left(q(z; \phi)\right) = \underbrace{\mathcal{L}(x;\theta, \phi)}_{\mathrm{ELBO}}\\

= & \mathcal{L}(\mathrm{x} ; \theta, \phi)+D_{K L}\left(q(\mathrm{z} ; \phi) \parallel p(\mathrm{z} | \mathrm{x} ; \theta)\right)


\end{align}
$$

더 좋은 $$q(z ; \phi)$$는 사후확률 $$p(z \vert x ; \theta)$$을 추정할 수 있고, 더 작은 $$D_{K L}\left(q(\mathrm{z} ; \phi) \parallel p(\mathrm{z} \vert \mathrm{x} ; \theta)\right)$$를 만들 수 있다. 우리는 더 작은 ELBO가 $$\log p(x ; \theta)$$가 됨을 알 수 있다.

그렇다면, 동시에 $$\theta$$와 $$\phi$$를 어떻게 최적화하여 주어진 데이터셋에 대해서 ELBO를 최대화 할 수 있을까?

### 중간 정리

지금까지 우리는 잠재 변수를 정의 했다. 

$$
p(x) \rightarrow p(x \mid z) \rightarrow p(x \mid z ; \theta)
$$

$$
p(x) \rightarrow p(x , z) \rightarrow p(x , z ; \theta)
$$

여기서 $$\sum_{\mathrm{x} \in \mathcal{D}} \log \sum_{\mathrm{z}} p(\mathrm{x}, \mathrm{z} ; \theta)$$ 를 얻기란 쉽지 않다. 

따라서 중요도 샘플링에 대해 제안하였고, ELBO를 정의하게 되었다. 

$$
\log p(\mathrm{x} ; \theta)=\sum_{\mathrm{z}} q(\mathrm{z}) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H(q)
$$

만약 $$p(z \mid x ; \theta)$$가 계산이 어렵다면? $$q(\mathrm{z} ; \phi)$$를 계산하기 쉽게 만들면 된다.

### Variational learning

[그림]

다시 정리해보면, $$\mathcal{L}\left(\mathrm{x} ; \theta, \phi_{1}\right)$$와 $$\mathcal{L}\left(\mathrm{x} ; \theta, \phi_{2}\right)$$는 둘 다 lower bound이고, 우리는 이 수식을 $$\theta$$와 $$\theta$$로 동시에 최적화를 수행하려 한다.

이제 이 아이디어를 전체 데이터셋으로 확장하면 된다.

ELBO는 어떠한 $$q(z; \phi)$$에 대해서도 만족한다.

$$
\log p(\mathrm{x} ; \theta) \geq \sum_{\mathrm{z}} q(\mathrm{z} ; \phi) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H(q(\mathrm{z} ; \phi))=\underbrace{\mathcal{L}(\mathrm{x} ; \theta, \phi)}_{\mathrm{ELBO}}
$$

따라서 전체 데이터셋에 대해서 maximal likelihood learning을 확장하면 다음과 같다.

$$
\ell(\theta ; \mathcal{D})=\sum_{x^{i} \in \mathcal{D}} \log p\left(x^{i} ; \theta\right) \geq \sum_{x^{i} \in \mathcal{D}} \mathcal{L}\left(x^{i} ; \theta, \phi^{i}\right)
$$

따라서,

$$
\max _{\theta} \ell(\theta ; \mathcal{D}) \geq \max _{\theta, \phi^{1}, \cdots, \phi^{M}} \sum_{x^{i} \in \mathcal{D}} \mathcal{L}\left(x^{i} ; \theta, \phi^{i}\right)
$$

여기서 우리는 다른 variational 파라미터 $$\phi^{i}$$를 모든 데이터 샘플 $$x^{i}$$에 대해 사용하게 된다. 그 이유는 우리가 진짜 사후확률 $$p\left(z \mid x^{i} ; \theta\right)$$이 각각의 데이터 포인트 $$x^{i}$$에 대해 다르기 때문이다.

### Learning via stochastic variational inference (SVI)

이제 우리는 $$\sum_{x^{i} \in \mathcal{D}} \mathcal{L}\left(x^{i} ; \theta, \phi^{i}\right)$$를 하나의 함수로서 $$\theta, \phi^{1}, \cdots, \phi^{M}$$ 를 하나의 stochastic gradient descent를 구하게 된다.

$$
\begin{aligned}
\mathcal{L}\left(\mathrm{x}^{i} ; \theta, \phi^{i}\right) & =\sum_{\mathrm{z}} q\left(\mathrm{z} ; \phi^{i}\right) \log p\left(\mathrm{z}, \mathrm{x}^{i} ; \theta\right)+H\left(q\left(\mathrm{z} ; \phi^{i}\right)\right) \\
& =E_{q\left(\mathrm{z} ; \phi^{i}\right)}\left[\log p\left(\mathrm{z}, \mathrm{x}^{i} ; \theta\right)-\log q\left(\mathrm{z} ; \phi^{i}\right)\right]
\end{aligned}
$$

이러한 최적화 과정은 다음 과정을 통해 수행된다.

- $$\theta, \phi^{1}, \cdots, \phi^{M}$$를 초기화한다.
- 데이터 포인트 $$x^{i}$$를 데이터셋 $$\mathcal{D}$$로 부터 얻는다.
- $$\phi^{i}$$의 함수 $$\mathcal{L}\left(x^{i} ; \theta, \phi^{i}\right)$$를 최적화 한다.
    - $$\phi^{i}=\phi^{i}+\eta \nabla_{\phi^{i}} \mathcal{L}\left(\mathrm{x}^{i} ; \theta, \phi^{i}\right)$$를 반복한다.
    - $$\phi^{i, *} \approx \arg \max _{\phi} \mathcal{L}\left(x^{i} ; \theta, \phi\right)$$가 수렴할때까지 반복한다.
- $$\nabla_{\theta} \mathcal{L}\left(x^{i} ; \theta, \phi^{i, *}\right)$$를 계산한다.
- $$\theta$$를 업데이트한다. 그런 다음 두번째 단계로 간다.
    - graident를 어떻게 계산하는가? 기대값을 계산하기에 정확한 다읍은 없기에 우리는 몬테카를로 샘플링을 한다.

### Learning Deep Generative Models

다시 정리해보면 

$$
\begin{aligned}
\mathcal{L}(\mathrm{x} ; \theta, \phi) & =\sum_{\mathrm{z}} q(\mathrm{z} ; \phi) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H(q(\mathrm{z} ; \phi)) \\
& =E_{q(\mathrm{z} ; \phi)}[\log p(\mathrm{z}, \mathrm{x} ; \theta)-\log q(\mathrm{z} ; \phi)]
\end{aligned}
$$

여기서 $$\phi^i$$의 superscript $$i$$는 제거하였다.

바운드를 계산하기 위해서 $$z^1, \cdots , z^k$$를 $$q(z;\phi)$$로 부터 샘플링 하고, 아래를 추정한다.

$$
E_{q(\mathrm{z} ; \phi)}\left[\log p(\mathrm{z}, \mathrm{x} ; \theta)-\log q(\mathrm{z} ; \phi)\right]
\approx \frac{1}{k}\sum_k \left( \log p(z^k,x;\theta)-\log q(z^k; \phi)\right)
$$

여기서 중요한 것은 $$q(z; \phi)$$이 계산 가능하다는 점이다. 즉 쉽게 샘플링하고 계산할 수 있는 분포이다.

우리가 $$\nabla_{\theta} \mathcal{L}(\mathrm{x} ; \theta, \phi)$$와 $$\nabla_{\phi} \mathcal{L}(\mathrm{x} ; \theta, \phi)$$를 계산할 수 있다면, gradient를 $$\theta$$에 대해 계산하는 것은 쉽다. 

$$
\begin{align}
\nabla_\theta E_{q(\mathrm{z} ; \phi)}[\log p(\mathrm{z}, \mathrm{x} ; \theta)-\log q(\mathrm{z} ; \phi)] 
= & E_{q(z;\phi)}\left[\nabla_\theta \log p(z,x;\theta)\right] \\
\approx & \frac{1}{k}\sum_k \nabla_\theta \log p(z^k, x; \theta)
\end{align}
$$

다만 $$\phi$$에 대한 graident를 계산하는것은 기대값이 $$\phi$$에 의존하기 때문에 좀 더 복잡하다. 하지만 우리는 여전이 몬테카를로 평균을 위해 추정을 하고 싶다. 현재로서는 더 좋지만 덜 일반화 가능한 기법은 연속 변수 $$z$$에 대해서 혹은 특정 분포에 대해서만 작동한다.

### Reparametrization

앞에서 언급한대로 $$\phi$$에 대한 graidient를 계산해보자.

$$
E_{q(\mathrm{z} ; \phi)}[r(\mathrm{z})]=\int q(\mathrm{z} ; \phi) r(\mathrm{z}) d \mathrm{z}
$$

이제 $$\mathrm{z}$$는 연속 랜덤 변수이다.

만약, $$q(\mathrm{z} ; \phi)=\mathcal{N}\left(\mu, \sigma^{2} I\right)$$가 가우시안이면서 파라미터 $$\phi=(\mu, \sigma)$$와 관련이 있다면 이것들은 샘플링을 다음과 같이 하는것과 같다.

$$\mathrm{z} \sim q_{\phi}(\mathrm{z})$$를 샘플링하고, $$\epsilon \sim \mathcal{N}(0, I), \mathrm{z}=\mu+\sigma \epsilon=g(\epsilon ; \phi)$$를 샘플링 하자. 그러면 아래 등식을 이용해 기대값을 두가지로 계산할 수 있다.

$$
\begin{align}
E_{z\sim q(z;\phi)}[r(z)] = & E_{\epsilon\sim\mathcal{N}(0, I)}[r(g(\epsilon;\phi))] = \int p(\epsilon)r(\mu + \sigma\epsilon)d\epsilon\\
\nabla_\phi E_{q(z;\phi)}[r(z)] = & \nabla_\phi E_\epsilon [r(g(\epsilon; \phi))] = E_\epsilon [\nabla_\phi r(g(\epsilon; \phi))]
\end{align}
$$

여기서 $$r$$과 $$g$$가 $$\phi$$에 대해 미분 가능하고, $$\epsilon$$가 샘플링이 쉬워진다면, 몬테카를로 추정이 쉬워진다. 

$$
E_{\epsilon}\left[\nabla_{\phi} r(g(\epsilon ; \phi))\right] \approx \frac{1}{k} \sum_{k} \nabla_{\phi} r\left(g\left(\epsilon^{k} ; \phi\right)\right)
\text{, where }
\epsilon^{1}, \cdots, \epsilon^{k} \sim \mathcal{N}(0, I)
$$
보통 이러한 기법은 REINFORCE보다는 훨씬 적은 variance를 가지게 된다.

이제 우리의 원래 수식을 살펴보자

$$
\begin{aligned}
\mathcal{L}(\mathrm{x} ; \theta, \phi) & =\sum_{\mathrm{z}} q(\mathrm{z} ; \phi) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H(q(\mathrm{z} ; \phi)) \\
& =E_{q(\mathrm{z} ; \phi)}[\underbrace{\log p(\mathrm{z}, \mathrm{x} ; \theta)-\log q(\mathrm{z} ; \phi)}_{r(\mathrm{z}, \phi)}]
\end{aligned}
$$

이 수식에서 우리는 $$E_{q(z ; \phi)}[r(z)]$$ 수식 대신, $$E_{q(\mathrm{z} ; \phi)}[r(\mathrm{z}, \phi)]$$를 가지고 있기 때문에 우리의 조건은 좀 더 복잡하다. 수식 안에서 기대값은 또한 $$\phi$$에 의존하고 있다.

여기서 우리는 여전히 reparameterization을 쓸 수 있다. 만약 $$\mathrm{z}=\mu+\sigma \epsilon=g(\epsilon ; \phi)$$이라 가정하면, 다음과 같다.

$$
\begin{aligned}
E_{q(\mathrm{z} ; \phi)}[r(\mathrm{z}, \phi)] & =E_{\epsilon}[r(g(\epsilon ; \phi), \phi)] \\
& \approx \frac{1}{k} \sum_{k} r\left(g\left(\epsilon^{k} ; \phi\right), \phi\right)
\end{aligned}
$$

### Amortized Inference

지금까지 우리는 다양한 셋의 variational parameter들 $$\phi^{i}$$의 각각 데이터 샘플 $$x^{i}$$에 대해 구해왔지만, 이러한 기법은 단순히 큰 데이터셋에 대해 확장하기 어렵다.

$$
\max _{\theta} \ell(\theta ; \mathcal{D}) \geq \max _{\theta, \phi^{1}, \cdots, \phi^{M}} \sum_{x^{i} \in \mathcal{D}} \mathcal{L}\left(x^{i} ; \theta, \phi^{i}\right)
$$

**Amortization**: 이제 우리는 파라미터를 가지는 **하나의 함수** $$f_{\lambda}$$를 가지고 있다고 생각하자. 그 함수는 각각의 $$x$$를 좋은 variational 파라미터로 매핑한다. 마치 regression을 하는 것과 같다.

$$x^{i} \mapsto \phi^{i, *}$$

예를 들어 $$q\left(z \mid x^{i}\right)$$가 다양한 평균 $$\mu^{1}, \cdots, \mu^{m}$$을 가지는 가우시안 함수라고 가정해보자. 그렇다면 우리는 하나의 인공 신경망 $$f_{\lambda}$$이 $$x^{i}$$에서 $$\mu^{i}$$로 가는 함수를 배우게 되는 것이다. 

이렇게 하면 우리는 사후확률 $$q\left(\mathrm{z} \mid \mathrm{x}^{i}\right)$$을 $$q_{\lambda}(\mathrm{z} \mid \mathrm{x})$$를 이용해서 추정할 수 있게 된다.

### A variational approximation to the posterior

[이진 영상 사진]

만약 $$p\left(z, x^{i} ; \theta\right)$$가 $$p_{\text {data }}\left(z, x^{i}\right)$$에 가깝다고 가정을 해보자. 그리고 잠재변수 $$z$$가 숫자, 스타일, 레벨과 같은 정보를 내재하고 있다고 가정해보자.

여기서 $$q\left(\mathrm{z} ; \phi^{i}\right)$$가 변수 $$z$$와 파라미터 $$\phi^{i}$$에 대해 계산하기 쉬운 (tractable)한 확률 분포라고 가정하자.

그렇다면 우리는 아래의 두가지 옵션을 가지게 된다.

- 각각의 데이터 샘플 $$x^{i}$$에 대해 $$\phi^{i, *}$$를 찾는다. (최적화를 이용할 수 있으나 매우 비싼 연산이다)
- 방금 배운 **Amortized inference**를 수행: 데이터 $$x^{i}$$가 좋은 파라미터의 집합 $$\phi^{i}$$으로 매핑할 수 있는 함수를 $$q\left(z ; f_{\lambda}\left(x^{i}\right)\right)$$를 이용해 구한다. $$f_{\lambda}$$는 최적화 문제를 어떻게 풀지를 알려준다.

참로고 $$q\left(z ; f_{\lambda}\left(x^{i}\right)\right)$$는 $$q_{\phi}(z \mid x)$$로도 표현된다.

이제 amortized inference를 이용해 학습을 해보자.

$$\sum_{x^{i} \in \mathcal{D}} \mathcal{L}\left(x^{i} ; \theta, \phi\right)$$를 $$\theta, \phi$$의 함수로 생각하여 stochastic gradient descent를 최적화 해보자. 

$$
\begin{aligned}
\mathcal{L}(\mathrm{x} ; \theta, \phi) & =\sum_{\mathrm{z}} q_{\phi}(\mathrm{z} | \mathrm{x}) \log p(\mathrm{z}, \mathrm{x} ; \theta)+H\left(q_{\phi}(\mathrm{z} | \mathrm{x})\right) \\
& \left.=E_{q_{\phi}(\mathrm{z} | \mathrm{x})}\left[\log p(\mathrm{z}, \mathrm{x} ; \theta)-\log q_{\phi}(\mathrm{z} | \mathrm{x})\right)\right]
\end{aligned}
$$

그렇다면 다음과 같은 단계를 생각해보자.

- $$\theta^{(0)}, \phi^{(0)}$$를 초기화 한다.
- 데이터 포인트 $$x^{i}$$를 데이터셋으로 부터 $$\mathcal{D}$$를 랜덤으로 샘플링 한다.
- $$\nabla_{\theta} \mathcal{L}\left(x^{i} ; \theta, \phi\right)$$와 $$\nabla_{\phi} \mathcal{L}\left(x^{i} ; \theta, \phi\right)$$를 계산한다.
- gradient 방향으로 $$\theta, \phi$$로 업데이트 한다.

그래디언트는 어떻게 계산할까? 이전에 배운 reparameterization을 쓰면 된다.

$$
\begin{aligned}
\mathcal{L}(\mathrm{x} ; \theta, \phi) & \left.=E_{q_{\phi}(\mathrm{z} \mid \mathrm{x})}\left[\log p(\mathrm{z}, \mathrm{x} ; \theta)-\log q_{\phi}(\mathrm{z} \mid \mathrm{x})\right)\right] \\
& \left.=E_{q_{\phi}(\mathrm{z} \mid \mathrm{x})}\left[\log p(\mathrm{z}, \mathrm{x} ; \theta)-\log p(\mathrm{z})+\log p(\mathrm{z})-\log q_{\phi}(\mathrm{z} \mid \mathrm{x})\right)\right] \\
& =E_{q_{\phi}(\mathrm{z} \mid \mathrm{x})}[\log p(\mathrm{x} \mid \mathrm{z} ; \theta)]-D_{K L}\left(q_{\phi}(\mathrm{z} \mid \mathrm{x}) \| p(\mathrm{z})\right)
\end{aligned}
$$

다시 정리해보면, 

- 데이터 포인트 $$x^{i}$$를 얻는다.
- $$\hat{z}$$ 가 $$q_{\phi}\left(z \mid x^{i}\right)$$로 매핑되도록 한다. (**인코더**에 해당한다)
- $$\hat{x}$$를 $$p(x \mid \hat{z} ; \theta)$$로 부터 복원한다. (**디코더**에 해당한다)

그렇다면 트레이닝 함수 $$\mathcal{L}(\mathrm{x} ; \theta, \phi)$$는 무엇을 할까? 
- 첫번째 항은 $$\hat{x} \approx x^{i}$$가 되는 것을 장려한다. ($$x^i$$가 $$p(x \mid \hat{z} ; \theta)$$에서 높도록 한다.) 
- 두번째 항은 $$\hat{z}$$가 $$p(z)$$항에서 높도록 한다.

### Learning Deep Generative models

이제 간단한 사례를 이용하여 지금까지 논의한 내용을 정리해보자.

- Alice가 우주 미션에 참여하게 되어서 이미지를 Bob에게 전달해야 한다. 이미지 $$x^{i}$$가 있을때, Alice는 이미지를 $$\hat{z} \sim q_{\phi}\left(z \mid x^{i}\right)$$를 이용해 압축을 하여 메시지 $$\hat{z}$$를 얻게 되고, 이를 Bob에게 전달한다.
- Bob는 $$\hat{z}$$를 가지고 $$p(x \mid \hat{z} ; \theta)$$를 이용해 원래의 이미지를 복원하려고 한다. 
    - 이러한 전략은 $$E_{q_{\phi}(\mathrm{z} \mid \mathrm{x})}[\log p(\mathrm{x} \mid \mathrm{z} ; \theta)]$$가 크다면 잘 작동할 것이다.
    - 수식 $$D_{K L}\left(q_{\phi}(\mathrm{z} \mid \mathrm{x}) \| p(\mathrm{z})\right)$$는 메시지에 대한 분포가 특별한 형태 $$p(z)$$가 되도록 강제한다. 만약 Bob가 $$p(z)$$에 대해 알고 있다면, Bob는 진짜같은 메시지를 $$\hat{z} \sim p(z)$$를 통해 얻을 수 있고, 이에 맞는 이미지도 생성할 수 있다. 마치 Alice에게 얻었던 것처럼 말이다.

### Summary of Latent Variable Models

지금까지 내용을 정리해보자.

- 간단한 모델을 결합하여 좀 더 유연한 기법을 만들 수 있다. (예를 들어 가우시안 합성 모델 처럼)
- Directed 모델은 ancesteral 샘플링을 가능하게 한다. (효율적인 생성이 가능하도록): $$\mathrm{z} \sim p(\mathrm{z}), \mathrm{x} \sim p(\mathrm{x} \mid \mathrm{z} ; \theta)$$
- 하지만, log-likelihood는 일반적으로 계산이 어렵다 (intractable), 따라서 학습은 어렵다.
- 모델 파라미터 $$(\theta)$$와 amortized inference component $$(\phi)$$는 ELBO최적화를 위해 계산 유용성을 향상시킨다.
- $$x$$를 위한 잠재적 표현은 $$q_{\phi}(\mathrm{z} \mid \mathrm{x})$$를 이용해서 얻을 수 있게 된다.

### 연구의 방향

따라서, 정리하자면, variational learning을 다음과 같이 올리는 것이 적절할 것이다.

- 더 좋은 최적화 기술
- 더 표현이 가능한 추정 기법들
- 대체할 수 있는 손실 함수

### 관련 연구들

**Model families - Encoder**

Amortization (Gershman & Goodman, 2015; Kingma; Rezende; ..)
- Scalability: Efficient learning and inference on massive datasets
- Regularization effect: Because of joint training, it also implicitly regularizes the model ￼ (Shu et al., 2018)

Augmenting variational posteriors
- Monte Carlo methods: Importance Sampling (Burda et al., 2015), MCMC (Salimans et al., 2015, Hoffman, 2017, Levy et al., 2018), Sequential Monte Carlo (Maddison et al., 2017, Le et al., 2018, Naesseth et al., 2018), Rejection Sampling (Grover et al., 2018)
- Normalizing flows (Rezende & Mohammed, 2015, Kingma et al., 2016) 

**Model families - Decoder**

- Powerful decoders ￼ such as DRAW (Gregor et al., 2015), PixelCNN (Gulrajani et al., 2016)
- Parameterized, learned priors ￼ (Nalusnick et al., 2016, Tomczak & Welling, 2018, Graves et al., 2018) 

**Variational objectives**

Tighter ELBO does not imply:

- Better samples: Sample quality and likelihoods are uncorrelated (Theis et al., 2016)
- Informative latent codes: Powerful decoders can ignore latent codes due to tradeoff in minimizing reconstruction error vs. KL prior penalty (Bowman et al., 2015, Chen et al., 2016, Zhao et al., 2017, Alemi et al., 2018)

Alternatives to KL divergence:
- Renyi's alpha-divergences (Li & Turner, 2016)
- Integral probability metrics such as maximum mean discrepancy, Wasserstein distance (Dziugaite et al., 2015; Zhao et. al, 2017; Tolstikhin et al., 2018)