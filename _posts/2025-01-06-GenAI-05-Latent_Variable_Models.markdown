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

