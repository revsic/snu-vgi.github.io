---
layout: post
title:  "GenAI Lecture: 04 Maximum Likelihood Learning"
date:   2025-01-05 01:09:12 +0900
categories: Gen AI Lecture
---

> 들어가기: 본 블로그 글은 S. Ermon, Y. Song, CS236 Deep Generative Models, Stanford University의 강의를 참고하여 제작되었습니다.

### Recap

[이미지 삽입]

다음과 같은 예시 데이터가 있을때 (강아지 사진 등), 우리는 이미지에 대한 확률 분포에 대해 이야기를 하고 있다. 이렇게 되면 얻을 수 있는 결과는 다음과 같다.

- 생성: $$x_\mathrm{new}\sim p(x)$$를 샘플링하면, $$x_\mathrm{new}\sim p(x)$$는 강아지의 모습일 것이다.
- 확률 예측: $$p(x)$$는 $$x$$가 강아지 처럼 보일 경우 높아야 한다. 그렇지 않으면 낮아야 한다. 
- 비지도 표현 학습: 이미지들로 부터 무엇이 공통적으로 있는지 (귀, 꼬리, 등의 특징) 알 수 있어야 한다.

그렇다면 첫번째 질문은 $$p(x)$$를 어떻게 표현할 것인지, 그리고 더 중요한것은 **어떻게 학습할 것인지**가 중요해진다.

### 학습에 대한 고찰

만약 데이터의 도메인이 특정 확률분포 $$P_{\mathrm{data}}$$에 결정된다고 가정하자.

그러면 주어진 데이터셋에 대해 $$\mathcal{D}$$ 샘플 $$m$$개를 $$P_{\mathrm{data}}$$로 부터 얻는것과 같아진다. 여기서 각 샘플은 각 변수의 할당에 해당한다. 예를 들면, $$(X_\mathrm{bank}=1, X_\mathrm{dollar}=0, \cdots, Y=1)$$ 혹은 픽셀 값 등

여기서 표준 가정은 데이터의 예시가 독립적이고 균일하게 분포가 되어 있다는 점이다. (independent and identically distributed (IID))

그리고 우리는 특정한 종류의 모델을 가지고 있다. $$\mathcal{M}$$ 이 중에서 우리는 좋은 모델 $$\hat{\mathcal{M}}\in \mathcal{M}$$을 얻어서 $$P_{\hat{\mathcal{M}}}$$을 표현 가능해야 한다. 예를 들면 모든 베이시안 네트워크는 CPD를 표로 표현 가능하다. 혹은 FVSBN에서는 모든 가능한 선택에 대해 logistric regression파라미터 $$\mathcal{M}=\{P_\theta, \theta\in\Theta\}$$에 대한 값이 필요하다. 편하게 정의하는 변수 $$\theta =$$ 모든 파라미터 값들을 이어 붙인 것으로 생각하자.

### 학습의 목적

이러한 경우 생성 모델을 위한 학습의 목표는 $$\hat{\mathcal{M}}$$이 분포 $$P_{\mathrm{data}}$$를 잘 표현하는 것이다. 

하지만 이러한 목적은 얻기가 쉽지 않은데, (1) 데이터가 부족해서 대략적인 분포를 알기 어려울 만큼 부족하거나 (2) 계산이 많은 이유가 있다.

예를 들어, 각 이미지를 784개의 이진 값으로 이어진 이미지라 생각해 보면 (백과 흑 픽셀) 가능한 경우의 수 혹은 가능한 이미지의 개수는 $$2^{784}\sim 10^{236}$$이다. 따라서, 이미지를 천만장 $$10^7 $$준비한다 하더라도 상당히 부족하다. 

따라서 우리는 $$\hat{\mathcal{M}}$$을 분포 $$P_{\mathrm{data}}$$에 대해 최선의 추정을 할 수 있는 모델을 선택해야 한다. 

여기서 최선의 모델은 어떤 것을 의미할까?

### 최선의 모델에 대하여

최선의 모델이란 우리가 무엇을 원하는지에 따라 달려 있다. 
1. 확률 밀도 함수 예측: 우리는 데이터를 표현하는 전체의 분포에 관심을 가지고 있다. (지금까지는 조건부 확률에 집중해 왔지만)
2. 구체적인 예측 task가 필요하다: 우리는 그 분포를 이용하여 예측을 한다. 예를 들어, 스팸 메일인지 아닌지, 다음 비디오에서 어떤 프레임이 나올지 등
3. 구조 혹은 지식을 발견하기: 이러할 경우 우리는 모델 자체에 관심을 갖게 될 것이다. 예를 들어, 유전자가 어떻게 서로 영향을 주는지, 무엇이 암을 유발하는지 등

### 밀도 추정을 학습의 목표로 둔다면

이러한 경우를 좀 더 생각해보면 우리가 전체 확률 분포를 학습하여 어떠한 확률 추론에 대한 질문에 대답을 할 수 있게 되는 것을 복표로 한다. 이러한 경우, 학습을 우리는 **밀도 추정**에 두게 된다. 이러한 경우 우리는 $$P_\theta$$를 최대한 $$P_{\mathrm{data}}$$에 가깝게 두는 것이 목표가 된다. (우리가 데이터셋 $$\mathcal{D}$$을 $$P_{\mathrm{data}}$$로 부터 얻은 것으로 가정하고 있음을 명심하자.)

그러면 두 확률 분포 $$P_\theta$$와 $$P_{\mathrm{data}}$$가 가깝다는 것을 어떻게 평가할 수 있을 것인가?

### KL-divergence

두개의 분포간의 거리를 측정하는 방법은 간단하다. Kullback-Leibler divergence (KL-divergence)는 분포 $$p$$와 $$q$$에 대해 거리를 다음과 같이 측정한다. 

$$
D(p\parallel q)=\sum_x p(x)log\frac{p(x)}{q(x)}
$$

여기서 $$D(p\parallel q)\geq 0$$는 모든 $$p, q$$에 대해 만족해야 한다. $$p=q$$라면 $$D$$가 $$0$$이 되는데 다음과 같이 쉽게 확인 가능하다

$$
E_{x\sim p}\left[ -\log \frac{q(x)}{p(x)} \right]\geq - \log \left(E_{x\sim p}\left[ \frac{q(x)}{p(x)} \right]\right) = -\log \left( \sum_x p(x) \frac{q(x)}{p(x)}\right)=0
$$

여기서 한가지 유의할 점은 KL-divergence는 비대칭이라는 점이다. 즉, $$D(p\parallel q)\neq D(q\parallel p)$$이다. 

### 데이터의 압축과 KL-divergence (여담)

압축을 하기 위해서는 데이터 샘플링에 사용한 분포를 아는 것이 매우 유용하다. 예를 들어 $$X_1, \cdots , X_{100}$$를 unbiased 동전이라고 정의를 해보자. 대략 50번의 앞면(Head)과 50번의 뒷면(Tail)이 나오게 된다. 가장 궁극적인 압축 기법은 당연히 앞면을 0, 뒷면을 1로 표현하는 것이 좋다. 이렇게 하면 샘플 당 1개의 bit를 표현하게 되고, 이보다 더 적은 bit를 활용하여 데이터를 표현할 수는 없다.

만약 동전이 biased되어 있다고 가정해보자. 예를 들면, $$P[H] \gg P[T]$$가 되어 있다고 가정해보자. 그렇다면, 당연히 앞면에 적은 비트를 사용하고 뒷면을 표현하기 위해 더 많은 비트를 사용하는 것이 더 현명한 방법이 된다. 

이러한 예시는 모스 부호에 나타나 있다. 이를 테면, 
```
E = •, A = • −, Q = − − • − 
```
와 같이 더 적게 사용하는 알파벳 q에 더 많은 bit를 할당하게 된다.

이와 같은 배경지식을 바탕으로 KL-divergence를 표현하면 어떻게 될까? 만약 데이터가 $$p$$의 분포에서 오는데, 현재 가지고 있는 표현법이 $$q$$로 표현이 된다고 하면, divergence $$D_{KL}(p\parallel q)$$는 평균적으로 필요한 추가의 bit수와 비례하게 된다.

### 밀도 추정을 학습의 목표로 둔다면 (계속)

앞선 목표에서 $$P_\theta$$를 최대한 $$P_\mathrm{data}$$에 가깝게 표현을 한다고 가정해보자 (데이터셋 $$\mathcal{D}$$는 $$P_\mathrm{data}$$의 샘플에서 얻게 된다.)

유사도는 어떻게 측정할 것인가?

이제는 KL-divergence가 하나의 해가 될 수 있음을 알 수 있다.

$$
D(P_{\mathrm{data}}\parallel P_\theta)=E_{x\sim P_{\mathrm{data}}}\left[ \log \left( \frac{P_{\mathrm{data}}(x)}{P_\theta (x)} \right) \right] = \sum_x P_{\mathrm{data}}(x)\log \frac{P_{\mathrm{data}}(x)}{P_\theta(x)}
$$

여기서 만약 두개의 분포가 같다면, $$D(P_\mathrm{data}\parallel P_\theta)=0$$가 된다. 앞선 여담을 다시 생각해보면, KL-divergence는 $$P_\mathrm{data}$$대신 $$P_\theta$$를 사용하면서 발생하게 되는 압축 손실 (bit)를 측정할 수 있게 된다.

### Expected log-likelihood

위의 수식을 좀 더 간단히 해보자. 

$$
\begin{align}
D(P_\mathrm{data}\parallel P_\theta ) & = E_{x\sim P_{\mathrm data}} \left[ \log \frac{P_\mathrm{data}(x)}{P_\theta (x)} \right]\\
& = E_{x\sim P_\mathrm{data}}[\log P_\mathrm{data}(x)]- E_{x\sim P_{\mathrm{data}}} [\log P_\theta (x)]
\end{align}
$$

첫번째 term은 $$P_\theta$$에 의존하지 않는다.

그렇다면, KL-divergence를 줄인다는 것은 곳 **expected log-likelihood**를 **최대화**한다는 것을 의미한다. 수식을 정리하면 다음과 같다.

$$
\begin{align}
\arg \min_{P_\theta} D(P_\mathrm{data} \parallel P_\theta) & = \arg \min_{P_\theta} - E_{x\sim P_\mathrm{data}} [\log P_\theta (x)] \\
& = \arg\max_{P_\theta}E_{x\sim P_\mathrm{data}} [\log P_\theta (x)]
\end{align}
$$

위의 수식은 $$P_\theta$$가 $$P_{\mathrm data}$$로 부터 얻은 샘플을 높은 확률 값으로 표시할 수 있음을 유도한다. 따라서 궁극적으로는 데이터의 분포를 표현하도록 도와준다. 그리고 위의 수식에서 $$\log $$함수의 특징으로 인해 샘플 $$x$$인 $$P_\theta(x)\sim 0$$는 위의 수식에서 상당한 가중치를 가지게 된다.

이제는 두개의 분포 사이의 유사도를 이론적으로 알 수 있게 되지만, 여기서 $$H(P_\mathrm{data})$$를 무시하고 있기 때문에, 최적의 결과까지 얼마나 떨어져 있는지는 알지 못한다. 즉, 문제는 $$P_{\mathrm data}$$를 모른다는 점이다.

### Maximum likelihood

이제 내용을 정리하여 expected log-likelihood를 추정하는 다음 수식을 생각해보자. 

$$
E_{x\sim P_\mathrm{data}}[\log P_\theta (x)]
$$

유한한 데이터에 대해 empirical log-likelihood는 다음과 같다.

$$
E_{\mathcal{D}}[\log P_\theta (x)] = \frac{1}{|\mathcal{D}|}\sum_{x\in\mathcal{D}}\log P_\theta (x)
$$

따라서 maximum likelihood 학습은 다음과 같이 정리된다.

$$
\max_{P_\theta}\frac{1}{|\mathcal{D}|}\sum_{x\in\mathcal{D}}\log P_\theta (x)
$$

등가적으로 우리는 다음의 likelihood를 최대화 하는것과 같다.

$$
P_\theta (x^{(1)}, \cdots, x^{(m)}) = \prod_{x\in\mathcal{D}}P_\theta(x)
$$

### 몬테카를로 추정의 아이디어

앞선 추정법은 유한한 값을 바탕으로 의미 있는 값을 추정하는 몬테카를로 추정과 연관이 된다. 아래 수식을 살펴보자. 

$$E_{x\sim P}[g(x)]=\sum_x g(x)P(x)$$

위의 수식은 $$T$$개의 샘플$$x^1, \cdots, x^T$$를 확률 분포 $$P$$에서 얻고 확률 분포에서 얻게 된다.

여기서 샘플의 기대 값(expected value)는 다음과 같이 얻을 수 있다. 

$$\hat{g}(x^1, \cdots, x^T) \triangleq \frac{1}{T}\sum_{t=1}^T g(x^t)$$

위의 수식에서 $$x^1, \cdots , x^T$$는 $$P$$로 부터 얻은 독립적인 샘플들이다. 

그리고 $$\hat{g}$$는 랜덤 변수이다. 왜 그럴까? 아래 몬테카를로 추정에 대한 세가지 재미있는 특징을 살펴보자.

- **Unbiased**: 기본적으로 몬테카를로 샘플 기법은 unbiased estimator이다. 

$$
E_{P}[\hat{g}]=E_{P}[g(x)]
$$

- **Convergence**: 큰 수의 법칙에 따라 다음을 만족한다.

$$
\hat{g}=\frac{1}{T} \sum_{t=1}^{T} g\left(x^{t}\right) \rightarrow E_{P}[g(x)] ~\text { for }~ T \rightarrow \infty
$$

- **Variance**: 얻은 샘플의 분산은 다음과 같다.

$$
V_{P}[\hat{g}]=V_{P}\left[\frac{1}{T} \sum_{t=1}^{T} g\left(x^{t}\right)\right]=\frac{V_{P}[g(x)]}{T}
$$

따라서 추정자의 분산은 샘플의 개수를 늘림에 따라 줄일 수 있다.

## 예시

하나의 변수 (바이어스 동전)에 대한 예시를 다시 생각해보자. 

- 이 랜덤 변수는 두가지 값을 가진다. 앞면 (Head) 및 뒷면 (Tail). 
- 동전을 던짐으로 인해 데이터셋을 얻게 된다. 예를 들어 5회 동전을 던져서 다음을 얻는다고 가정하자. $$\mathcal{D}=\{H, H, T, H, T\}$$
- 모델의 종류 $$\mathcal{M}$$: 랜덤 변수 $$x \in\{H, T\}$$를 표현할 수 있는 모든 확률 분포를 생각해보자.

그렇다면, 만약 100번 동전 던지기에서 60번의 head가 나오는 데이터셋 $$\mathcal{D}$$를 가진다면, 어떻게 $$P_{\theta}(x)$$를 $$\mathcal{M}$$으로 부터 얻을 수 있을까?

다음과 같은 모델을 생각해보자.

$$P_{\theta}(x=H)=\theta$$

$$P_{\theta}(x=T)=1-\theta$$

그러면 가지고 있는 데이터셋을 가지고 likelihood를 표현하면 다음과 같다. 

$$=\prod_{i} P_{\theta}\left(x_{i}\right)=\theta \times \theta \times(1-\theta) \times \theta \times(1-\theta)$$

[그림]

그렇다면 파라미터 $$\theta$$ 중에서 데이터셋 $$\mathcal{D}$$에 대해 가장 높은 likelihood를 갖게 하는 값은 무엇일까? 그림에서 보면 알 수 있듯, 0.6인 것을 알 수 있다.

수식을 유도해보자. 좀 더 일반적으로 log-likelihood 함수를 얻으면 다음과 같아진다. 

$$
\begin{aligned}
L(\theta) & =\theta^{\# \text {heads }} \times(1-\theta)^{\# \text { tails }} \\
\log L(\theta) & =\log \left(\theta^{\# \text { heads }} \times(1-\theta)^{\# \text { tails }}\right) \\
& =\# \text { heads } \times \log (\theta)+\# \text { tails } \times \log (1-\theta)
\end{aligned}
$$

이제 MLE의 목표는 $$\log L\left(\theta^{*}\right)$$를 최대화하는 값 $$\theta^{*} \in[0,1]$$를 찾는 것이다. 이제 $$\theta$$에 대해 미분하고, 미분한 값을 0으로 두면, 다음을 얻는다. 

$$
\theta^{*}=\frac{\# \text { heads }}{\# \text { heads }+\# \text { tails }}
$$

### MLE 아이디어를 autoregressive models로 확장하기

변수 $$n$$개로 표현되고 factorize되는 autoregressive 모델이 주어져 있다고 가정하자.

$$
P_{\theta}(\mathrm{x})=\prod_{i=1}^{n} p_{\text {neural }}\left(x_{i} \mid \mathrm{x}_{<i} ; \theta_{i}\right)
$$

$$\theta=\left(\theta_{1}, \cdots, \theta_{n}\right)$$는 모든 조건에 대한 파라미터들이다.

학습 데이터 $$\mathcal{D}=\left\{\mathrm{x}^{(1)}, \cdots, \mathrm{x}^{(m)}\right\}$$가 주어져 있을때 파라미터 $$\theta$$에 대한 maximum liklihood 추정값은 무엇일까? 

likelihood function을 분리하면 다음과 같다.

$$
L(\theta, \mathcal{D})=\prod_{j=1}^{m} P_{\theta}\left(x^{(j)}\right)=\prod_{j=1}^{m} \prod_{i=1}^{n} p_{\text {neural }}\left(x_{i}^{(j)} \mid x_{<i}^{(j)} ; \theta_{i}\right)
$$

이제 우리의 목적은 $$\arg \max _{\theta} L(\theta, \mathcal{D})=\arg \max _{\theta} \log L(\theta, \mathcal{D})$$를 얻는 것이고, 더이상 닫힌 해를 얻을 수 없게 된다. 

### MLE Learning: Gradient Descent

아래의 수식이 주어져 있을때, 

$$
L(\theta, \mathcal{D})=\prod_{j=1}^{m} P_{\theta}\left(x^{(j)}\right)=\prod_{j=1}^{m} \prod_{i=1}^{n} p_{\text {neural }}\left(x_{i}^{(j)} \mid x_{<i}^{(j)} ; \theta_{i}\right)
$$

$$\arg \max _{\theta} L(\theta, \mathcal{D})=\arg \max _{\theta} \log L(\theta, \mathcal{D})$$를 하기 위해서는 어떤걸 해야 할까?

$$
\ell(\theta)=\log L(\theta, \mathcal{D})=\sum_{j=1}^{m} \sum_{i=1}^{n} \log p_{\text {neural }}\left(x_{i}^{(j)} \mid x_{<i}^{(j)} ; \theta_{i}\right)
$$

다음과 같은 방법을 고려해보자.

1. $$\theta^{0}=\left(\theta_{1}, \cdots, \theta_{n}\right)$$를 랜덤으로 초기화 한다.
2. 역전파(back propagation)를 활용해 $$\nabla_{\theta} \ell(\theta)$$ 를 얻는다.
3. $$\theta^{t+1}=\theta^{t}+\alpha_{t} \nabla_{\theta} \ell(\theta)$$를 이용해 파라미터를 업데이트 한다.

위의 접근법은 non-convex 최적화 문제를 바탕으로 하고 있다. 하지만 실제로는 꽤나 잘 작동한다.

### MLE Learning: Stochastic Gradient Descent

그렇다면 $$\theta_{i}$$의 gradient는 어떻게 구해질까?

$$
\nabla_{\theta_{i}} \ell(\theta)=\sum_{j=1}^{m} \nabla_{\theta_{i}} \sum_{i=1}^{n} \log p_{\text {neural }}\left(x_{i}^{(j)} \mid x_{<i}^{(j)} ; \theta_{i}\right)=\sum_{j=1}^{m} \nabla_{\theta_{i}} \log p_{\text {neural }}\left(x_{i}^{(j)} \mid x_{<i}^{(j)} ; \theta_{i}\right)
$$

각각의 조건부 확률 $$p_{\text {neural }}\left(x_{i} \mid \mathrm{x}_{<i} ; \theta_{i}\right)$$는 파라미터 공유가 되어 있지 않으면 각각 최적화 될 수 있다. 실제로는 많은 경우 파라미터 $$\theta_{i}$$는 공유된다. (예: NADE, PixelRNN, PixelCNN 등등) 

만약 $$m=\vert\mathcal{D}\vert$$가 매우 커진다면 어떨까?

$$
\begin{align}
\nabla_\theta \ell(\theta) = & m \sum_{j=1}^{m}\frac{1}{m}\sum_{i=1}^{n}\nabla_\theta \log p_{\text{neural}} (x_i^{(j)} | x_{<i}^{(j)}; \theta_i)\\
\quad\quad\quad = & mE_{x^{(j)}\sim\mathcal{D}} \left[ \sum_{i=1}^{n}\nabla_\theta \log p_{\text{neural}} (x_i^{(j)} | x_{<i}^{(j)};\theta_i) \right]
\end{align}
$$

이제 몬테카를로 기법을 쓸 수 있다. 

$$
x^{(j)} \sim \mathcal{D} ; \nabla_{\theta} \ell(\theta) \approx m \sum_{i=1}^{n} \nabla_{\theta} \log p_{\text {neural }}\left(x_{i}^{(j)} \mid x_{<i}^{(j)} ; \theta_{i}\right)
$$

### Empirical Risk and Overfitting

[TBA]

### 요약

autoregressive model들에 대해 $$p_{\theta}(x)$$의 계산은 쉽다. 이상적으로는 RNN과 달리 각각의 조건부 확률 $$\log p_{\text {neural }}\left(x_{i}^{(j)} \mid \mathrm{x}_{<i}^{(j)} ; \theta_{i}\right)$$도 얻을 수 있다. 
다만, 이러한 기법은 maximum likelihood와도 이어지고 더 큰 log-likelihood는 항상 더 좋아 보이는 샘플을 의미하는 것은 아니다. 다른 방식으로 유사도를 측정하는 방법도 있다. 예를 들면 Generative Adversarial Networks등이 있다. 