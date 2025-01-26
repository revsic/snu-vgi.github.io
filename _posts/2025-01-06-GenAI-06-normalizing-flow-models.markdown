---
layout: post
title:  "GenAI Lecture: 06 Normalizing Flow Models"
date:   2025-01-05 01:09:12 +0900
categories: Gen AI Lecture
---

> 들어가기: 본 블로그 글은 S. Ermon, Y. Song, CS236 Deep Generative Models, Stanford University의 강의를 참고하여 제작되었습니다.

### Recap

지금까지 우리가 배운 likelihood기반 학습 기법들은 다음과 같다.

- 모델의 종류
    - Autoregressive 모델: $$p_{\theta}(\mathrm{x})=\prod_{i=1}^{n} p_{\theta}\left(x_{i} \mid \mathrm{x}_{<i}\right)$$
    - variational autoencoder: $$p_{\theta}(\mathrm{x})=\int p_{\theta}(\mathrm{x}, \mathrm{z}) d \mathrm{z}$$
- Autoregressive 모델은 계산이 편한 likelihood를 제공하지만 직접적으로 특징을 배울 직접적인 메커니즘을 갖고 있지는 않다. 
- Variational autoencoder는 특징 표현을 배울 수 있지만 (잠재변수 $$z$$를 이용해서) 하지만 직접 계산하기 어려운 marginal likelihood를 갖고 있다. 
**주요 질문** 더 좋은 잠재 함수 모델을 계산하기 쉬운 likelihood를 이용해 배울 수 있을까? 가능하다!

### Simple Prior to Complex Data Distributions

$$p_{\theta}(\mathrm{x})$$의 모델을 표현하기 위해 기대되는 특징은 무엇이 있을까? 
- 계산이 쉬워야 한다. 닫힌 형태의 확률 분포를 가진다. (학습시 유리하다)
- 샘플링이 쉬워야 한다. (생성에 유용하다)

간단한 분포는 위의 특징을 만족한다. 예) 가우시안 함수, 균등분포 함수 등. 하지만 데이터의 분포는 좀 더 복잡하다 (멀티 모달 데이터 등) 

**Flow모델에 대한 핵심 아이디어**: 간단한 분포를 역변환 가능 변환식을 이용해 복잡한 분포로 변환하는 것이 핵심 아이디어이다.

### Variational Autoencoder

이러한 플로우 모델의 아이디어는 variational autoencoder와 유사하다.

- 간단한 prior함수 $$\mathrm{z} \sim \mathcal{N}(0, I)=p(\mathrm{z})$$로 부터 시작한다.
- $$p(\mathrm{x} \mid \mathrm{z})=\mathcal{N}\left(\mu_{\theta}(\mathrm{z}), \Sigma_{\theta}(\mathrm{z})\right)$$수식을 이용해 변환을 한다.
- 만약 $$p(z)$$가 간단하더라도, $$p_{\theta}(x)$$는 매우 복잡하고 유연하다. 하지만 $$p_{\theta}(\mathrm{x})=\int p_{\theta}(\mathrm{x}, \mathrm{z}) d \mathrm{z}$$는 계산하기 매우 비싸다. 모든 변수 $$x$$를 만들 수 있는 모든 $$z$$에 대하여 고려해야 하기 때문이다.
- 만약 우리가 간단하게 $$p(\mathrm{x} \mid \mathrm{z})$$를 역변환하고, $$p(\mathrm{z} \mid \mathrm{x})$$를 쉽게 계산할 수 있도록 하면 어떨까? $$\mathrm{x}=f_{\theta}(\mathrm{z})$$를 deterministic하고 $$z$$에 대한 역변환 가능한 함수로 정의하면 될 것이다!

