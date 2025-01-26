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

### Continuous random variables refresher

$$X$$를 연속적인 랜덤 변수라고 생각해보자. $$X$$의 누적 분포 함수 (cumulative density function: CDF)는 $$F_{X}(a)=P(X \leq a)$$가 되고, $$X$$의 확률 밀도 함수는 $$p_{X}(a)=F_{X}^{\prime}(a)=\frac{d F_{X}(a)}{d a}$$이다.

일반적으로 파라미터로 표현할 수 있는 확률 밀도 함수는 다음과 같다.
- 가우시안: 만약 $$p_{X}(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-(x-\mu)^{2} / 2 \sigma^{2}}$$이라면, $$X \sim \mathcal{N}(\mu, \sigma)$$이다.
- 균일분포함수: $$p_{X}(x)=\frac{1}{b-a} 1[a \leq x \leq b]$$이고, $$X \sim \mathcal{U}(a, b)$$로 표현한다.
- 이 밖의 다른 함수들도 있다.

만약 $$\boldsymbol{X}$$가 연속적인 랜덤 벡터일 경우 우리는 결합 확률 밀도 함수로 표현 가능하다. 예를 들어, 가우시안일 경우 아래와 같다.

$$
p_{X}(\mathrm{x})=\frac{1}{\sqrt{(2 \pi)^{n}|\Sigma|}} \exp \left(-\frac{1}{2}(\mathrm{x}-\boldsymbol{\mu})^{T} \Sigma^{-1}(\mathrm{x}-\boldsymbol{\mu})\right)
$$

### Change of Variables formula

$$Z$$가 균등 분포함수의 랜덤변수 $$\mathcal{U}[0,2]$$라 하고, 밀도 함수 $$p_{Z}$$라 했을때, $$p_{Z}(1)$$은 무엇일까? $$\frac{1}{2}$$이다. 검산을 해보면 $$\int_{0}^{2} \frac{1}{2}=1$$이다.

만약 새로운 랜덤 변수 $$X=4 Z$$를 정의하고, $$p_{X}$$를 밀도함수라고 정하면, $$p_{X}(4)$$는 무엇일까?

보통은 다음과 같이 생각할 것이다.

$$
p_{X}(4)=p(X=4)=p(4 Z=4)=p(Z=1)=p_{Z}(1)=\frac{1}{2}
$$

하지만 이는 잘못된 답변이다. 명백하게도 $$X$$는 $$[0,8]$$에서 균등 분포함수 이고, $$p_{X}(4)=\frac{1}{8}$$이다. 따라서 정확한 답을 얻기 위해서는 **change of variables formula**가 필요하다.

**1차원 변수일 경우**: 만약 $$X=f(Z)$$이고, $$f(\cdot)$$가 단조함수이며, 역함수가 $$Z=f^{-1}(X)=h(X)$$라고 했을때, 다음 수식을 만족한다.

$$
p_{X}(x)=p_{Z}\left(h(x)\right)\left|h^{\prime}(x)\right|
$$

이전의 예제를 돌아보면, $$X=f(Z)=4 Z$$이고 $$Z \sim \mathcal{U}[0,2]$$라면, $$p_{X}(4)$$는 무엇인가?

$$h(X)=X / 4$$임을 상기하자. 그러면, $$p_{X}(4)=p_{Z}(1) h^{\prime}(4)=\frac{1}{2} \times \mid \frac{1}{4}\mid=\frac{1}{8}$$이다.

좀 더 재미있는 예시를 들어보자. 만약 $$X=f(Z)=\exp (Z)$$이고, $$Z \sim \mathcal{U}[0,2]$$라면, $$p_{X}(x)$$는 무엇인가?

$$h(X)=\ln (X)$$임을 명시하자. 그렇다면, $$x \in[\exp (0), \exp (2)]$$에 대해 $$p_{X}(x)=p_{Z}(\ln (x))\vert h^{\prime}(x)\vert=\frac{1}{2 x}$$이다.
여기서 $$p_{X}(x)$$의 모양은 사전함수 $$p_{Z}(z)$$와 다르다는 점을 기억하자. 

증명에 대해 생각해보자. 만약 $$f(\cdot)$$가 단조 증가함수일 경우에는 

$$
F_{X}(x)=p[X \leq x]=p[f(Z) \leq x]=p[Z \leq h(x)]=F_{Z}(h(x))
$$

만약에 양변에 미분을 한다면:

$$
p_{X}(x)=\frac{d F_{X}(x)}{d x}=\frac{d F_{Z}(h(x))}{d x}=p_{Z}\left(h(x)\right) h^{\prime}(x)
$$

만약 기본적인 대수학을 생각해보면 

$$
h^{\prime}(x)=\left[f^{-1}\right]{\prime}(x)=\frac{1}{f{\prime}\left(f^{-1}(x)\right)}
$$

따라서, $$z=h(x)=f^{-1}(x)$$로 두면, 다음과 같다.

$$
p_{X}(x)=p_{Z}(z) \frac{1}{f^{\prime}(z)}
$$

### Geometry: Determinants and volumes

만약 $$Z$$가 균등 분포 벡터 $$[0,1]^{n}$$라고 정의해보자. 그리고 $$X=A Z$$가 square 역행렬 $$W=A^{-1}$$을 가지는 행렬 $$A$$로 정의되어 있다고 생각해보자. 그렇다면, $$X$$는 어떻게 분포되어 있을까?

기하적으로, 행렬 $$A$$는 균등 하이퍼큐브 $$[0,1]^{n}$$를 평행체로 변형한다고 생각해보자. 하이퍼큐브와 평행체는 정사각형/큐브와 평행사변형/평행체로 대응되는 고차원으로 일반화가 될 수 있다.

[그림 삽입]

이러한 조건에서 평행체의 부피는 흥미롭게도 행렬 $$A$$의 행렬 계수 (determinant)가 된다.

$$
\det (A) 
= \det \begin{pmatrix}
a & c\\
b & d
\end{pmatrix}	
= ad - bc
$$

이럴때, $$X=A Z$$를 square의 역행렬 $$W=A^{-1}$$ 이 있는 행렬 $$A$$로 가정하면, $$X$$는 평행체에 대해 면적 $$\vert \operatorname{det}(A)\vert$$로 균일하게 분포되어 있다. 따라서, 우리는 다음을 얻는다.

$$
\begin{aligned}
p_{X}(\mathrm{x}) & =\frac{p_{Z}(W \mathrm{x})}{|\operatorname{det}(A)|} \\
& =p_{Z}(W \mathrm{x})|\operatorname{det}(W)|
\end{aligned}
$$

$$W=A^{-1}, \operatorname{det}(W)=\frac{1}{\operatorname{det}(A)}$$이기 때문이다. 여기서 같는 1차원 수식의 유사도를 확인하자.

### Generalized change of variables

정리하면, 선형 변환 $$A$$에 대해서 부피의 변화는 $$A$$의 행렬 계수값으로 결정된다.

만약 비선형 변환 $$\mathrm{f}(\cdot)$$에 대해서라면, 부피의 선형적 변화는 $$\mathrm{f}(\cdot)$$의 자코비안의 행렬 계수로 주어지게 된다.

**Change of variables (General case)** 변수 $$Z$$와 $$X$$의 매핑 관계에서 $$f: \mathbb{R}^{n} \mapsto \mathbb{R}^{n}$$, 이것은 역변환이 가능하다. 즉, $$X=\mathrm{f}(Z)$$이고, $$Z=\mathrm{f}^{-1}(X)$$이다.

$$
p_{X}(\mathrm{x})=p_{Z}\left(\mathrm{f}^{-1}(\mathrm{x})\right)\left|\operatorname{det}\left(\frac{\partial \mathrm{f}^{-1}(\mathrm{x})}{\partial \mathrm{x}}\right)\right|
$$

- 이 수식은 1차원 사례를 일반화 한다. $$p_{X}(x)=p_{Z}(h(x))\mid h^{\prime}(x)\mid$$
- VAE들과 달리 $$x, z$$는 연속이어야 하고, **같은 차원**이어야 한다. 즉, $$\mathrm{x} \in \mathbb{R}^{n}$$이면, $$\mathrm{z} \in \mathbb{R}^{n}$$이다.
- 역변환이 가능한 함수 $$A, \operatorname{det}\left(A^{-1}\right)=\operatorname{det}(A)^{-1}$$에 대해서, 다음을 만족한다.

$$
p_{X}(\mathrm{x})=p_{Z}(\mathrm{z})\left|\operatorname{det}\left(\frac{\partial \mathrm{f}(\mathrm{z})}{\partial \mathrm{z}}\right)\right|^{-1}
$$

### Two Dimensional Example

$$Z_{1}$$와 $$Z_{2}$$를 연속적인 랜덤 변수이고, 결합 확률 분포 $$p_{Z_{1}, Z_{2}}$$를 가지고 있다고 생각하자. 여기서, $$u=\left(u_{1}, u_{2}\right)$$는 변환이고, $$v=\left(v_{1}, v_{2}\right)$$는 역변환이다. 

$$X_{1}=u_{1}\left(Z_{1}, Z_{2}\right)$$이고, $$X_{2}=u_{2}\left(Z_{1}, Z_{2}\right)$$이라면, $$Z_{1}=v_{1}\left(X_{1}, X_{2}\right)$$이고, $$Z_{2}=v_{2}\left(X_{1}, X_{2}\right)$$이다. 

따라서, 다음과 같이 정리된다. 

$$
\begin{align}
p_{X_1, X_2} (x_1, x_2) = & p_{Z_1, Z_2}\left( v_1(x_1, x_2), v_2(x_1, x_2) \right)

\begin{vmatrix}

\frac{\partial v_1(x_1, x_2)}{\partial x_1} & \frac{\partial v_1 (x_1, x_2)}{\partial x_2} \\
\frac{\partial v_2 (x_1, x_2)}{\partial x_1} & \frac{\partial v_2 (x_1, x_2)}{\partial x_2}

\end{vmatrix} (역변환)\\

= & p_{Z_1, Z_2}(z_1, z_2) \begin{vmatrix}
\frac{\partial v_1(x_1, x_2)}{\partial x_1} & \frac{\partial v_1 (x_1, x_2)}{\partial x_2} \\
\frac{\partial v_2 (x_1, x_2)}{\partial x_1} & \frac{\partial v_2 (x_1, x_2)}{\partial x_2}

\end{vmatrix}^{-1} (정변환)

\end{align}
$$


### Normalizing flow models

이제 이야기한 내용들을 정리해보자. 만약 방향정을 가진 잠재 변수 모델을 관측한 변수 $$X$$와 $$Z$$에 대해 정의해보자.

Normalizing flow model에서, 변수 $$Z$$와 $$X$$에 대해 함수 $$\mathrm{f}_{\theta}: \mathbb{R}^{n} \mapsto \mathbb{R}^{n}$$는 determinstic이고 역변환이 $$X=\mathrm{f}_{\theta}(Z)$$이고 $$Z=\mathrm{f}_{\theta}^{-1}(X)$$로 가능하다. 

그러면 change of variables의 수식을 활용하여 marginal likelihood는 $$p(\mathrm{x})$$ 다음과 같이 주어진다.

$$
p_{X}(\mathrm{x} ; \theta)=p_{Z}\left(\mathrm{f}_{\theta}^{-1}(\mathrm{x})\right)\left|\operatorname{det}\left(\frac{\partial \mathrm{f}_{\theta}^{-1}(\mathrm{x})}{\partial \mathrm{x}}\right)\right|
$$

여기서 $$\mathrm{x}, \mathrm{z}$$는 연속적이고 같은 차원을 가져야 한다는 점을 기억하자.

### A Flow of Transformations

**Normalizing**: 변수의 변환은 역변환이 가능한 함수를 활용해, 정규화된 분포를 가져다 준다.

**Flow** 역변환이 가능한 변화는 다음과 같이 결합된다. 

$$
\mathrm{z}_{m}=\mathrm{f}_{\theta}^{m} \circ \cdots \circ \mathrm{f}_{\theta}^{1}\left(\mathrm{z}_{0}\right)=\mathrm{f}_{\theta}^{m}\left(\mathrm{f}_{\theta}^{m-1}\left(\cdots\left(\mathrm{f}_{\theta}^{1}\left(\mathrm{z}_{0}\right)\right)\right)\right) \triangleq \mathrm{f}_{\theta}\left(\mathrm{z}_{0}\right)
$$

따라서 간단한 분포 $$\mathrm{z}_{0}$$ (예를 들면 가우시안)으로 시작하고, $$M$$개의 역변환이 가능한 변환을 한 다음에 최종적으로 $$\mathrm{x}=\mathrm{z}_{\mathrm{M}}$$를 얻으면 된다. 

$$
p_{X}(\mathrm{x} ; \theta)=p_{Z}\left(\mathrm{f}_{\theta}^{-1}(\mathrm{x})\right) \prod_{m=1}^{M}\left|\operatorname{det}\left(\frac{\partial\left(\mathrm{f}_{\theta}^{m}\right)^{-1}\left(\mathrm{z}_{m}\right)}{\partial \mathrm{z}_{m}}\right)\right|
$$

행렬들의 곱의 행렬 계수는 행렬 계수의 곱과 같다.

[변환 수식 그림]

다음과 같은 예시를 볼 수 있다. 기준 모델 (가우시안), 기준 모델 (균일함수). 10개의 선형 변환은 간단한 확률 분포를 훨신 복잡한 분포로 변환할 수 있다.

### 학습과 추론

**maximal likelihood**로 데이터셋 $$\mathcal{D}$$을 이용해 학습하면 다음과 같다.

$$
\max _{\theta} \log p_{X}(\mathcal{D} ; \theta)=\sum_{\mathrm{x} \in \mathcal{D}} \log p_{Z}\left(\mathrm{f}_{\theta}^{-1}(\mathrm{x})\right)+\log \left|\operatorname{det}\left(\frac{\partial \mathrm{f}_{\theta}^{-1}(\mathrm{x})}{\partial \mathrm{x}}\right)\right|
$$

다음과 같은 재미있는 특징을 지닌다.
- 정확한 likelihood 값 계산은 역변환 $$\mathrm{x} \mapsto \mathrm{z}$$과 change of variable 공식으로 구할 수 있다.
- 샘플링은 다음과 같은 변환을 통해 얻는다. $$\mathrm{z} \mapsto \mathrm{x}$$

$$
\mathrm{z} \sim p_{Z}(\mathrm{z}) \quad \mathrm{x}=\mathrm{f}_{\theta}(\mathrm{z})
$$

- 잠재 변수의 표현은 다음과 같은 역변환함수를 이용하여 가능하다 (inference를 위한 네트워크가 필요하지 않다.)

$$
\mathrm{z}=\mathrm{f}_{\theta}^{-1}(\mathrm{x})
$$


### Desiderata for flow models

이처럼 간단한 사전확률 $$p_{Z}(\mathrm{z})$$를 이용하면, 효율적인 샘플링과 계산이 편한 likelihood 값의 계산이 가능하다. 

역변환이 가능한 변환은 손쉽게 계산이 가능하다.
- likelihood를 계산하기 위해 효과적인 $$\mathrm{x} \mapsto \mathrm{z}$$의 계산이 필요하다.
- 샘플링은 효과적인 $$\mathrm{z} \mapsto \mathrm{x}$$ 매핑을 필요로 한다.

likelihood를 계산하는 것 또한 $$n \times n$$크기의 자코비안 행렬의 행렬 계수 (determinant)를 구하는 것을 요구한다. 여기서 $$n$$은 데이터의 차원을 의미한다.
- 다시 말하자면 행렬 계수 $$n \times n$$ 행렬에 대한 값을 구하는 것은 $$O\left(n^{3}\right)$$의 연산량이 필요하고, 학습 루프에 있는 것은 피해야만 할 것이다.
- 핵심 아이디어: 변환식을 특별하게 두어서 자코비안 행렬이 특별한 구조를 갖게 한다. 예를 들면 삼각행렬 (triangular matrix)는 대각에 있는 값들을 곱하는 것으로 손쉽게 행렬 계수를 구할 수 있다. 즉, $$O(n)$$의 계산만이 필요하다.

$$
\mathrm{x}=\left(x_{1}, \cdots, x_{n}\right)=\mathrm{f}(\mathrm{z})=\left(f_{1}(\mathrm{z}), \cdots, f_{n}(\mathrm{z})\right)
$$

이라면, 다음과 같다. 

$$
J=\frac{\partial \mathrm{f}}{\partial \mathrm{z}}=\begin{pmatrix}
\frac{\partial f_{1}}{\partial z_{1}} & \cdots & \frac{\partial f_{1}}{\partial z_{n}} \\
\cdots & \cdots & \cdots \\
\frac{\partial f_{n}}{\partial z_{1}} & \cdots & \frac{\partial f_{n}}{\partial z_{n}}
\end{pmatrix}
$$

만약 $$x_{i}=f_{i}(\mathrm{z})$$가 $$\mathrm{z}_{\leq i}$$에만 의존한다고 가정해보자. 그렇다면,

$$
J=\frac{\partial f}{\partial z}=\begin{pmatrix}
\frac{\partial f_{1}}{\partial z_{1}} & \cdots & 0 \\
\cdots & \cdots & \cdots \\
\frac{\partial f_{n}}{\partial z_{1}} & \cdots & \frac{\partial f_{n}}{\partial z_{n}}
\end{pmatrix}
$$

lower 삼각행렬을 같게 된다. 행렬 계수는 선형적으로 계산이 가능하다. 유사하게 $$x_{i}$$가 $$z_{\geq i}$$에만 의존한다면, 자코비안 함수는 upper 삼각행렬이다.

### Planar flows (Rezende & Mohamed, 2016)

Planar flow는 다음과 같은 역변환을 가진다. 

$$
\mathrm{x}=\mathrm{f}_{\theta}(\mathrm{z})=\mathrm{z}+\mathrm{u}h\left(\mathrm{w}^{T} \mathrm{z}+b\right)
$$

이 수식은 $$\theta=(\mathrm{w}, \mathrm{u}, b)$$의 파라미터로 표현이 가능하고, $$h(\cdot)$$는 비선형을 나타낸다. 

자코비안 행렬의 행렬계수 절대값은 다음과 같이 주어진다.

$$
\begin{align}

\left|\operatorname{det} \frac{\partial \mathrm{f}_{\theta}(\mathrm{z})}{\partial \mathrm{z}}\right|= & \left|\operatorname{det}\left(I+h^{\prime}\left(\mathrm{w}^{T} \mathrm{z}+b\right) \mathrm{uw}^{T}\right)\right| \\

= & \left|1+h^{\prime}\left(\mathrm{w}^{T} \mathrm{z}+b\right) \mathrm{u}^{T} \mathrm{w}\right|

\end{align}
$$

(행렬 계수의 보조정리에 의해). 이런 방식으로 파라미터를 제한하거나, 역변환이 가능하도록 비선형성을 정의한다. 예를 들어서 $$h= tanh()$$이고, $$h^{\prime}\left(\mathrm{w}^{T} \mathrm{z}+b\right) \mathrm{u}^{T} \mathrm{w} \geq-1$$이다.