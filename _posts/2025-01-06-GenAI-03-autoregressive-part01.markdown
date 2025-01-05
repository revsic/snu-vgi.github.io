---
layout: post
title:  "GenAI Lecture: 03 Auto Regressive Models"
date:   2025-01-05 01:09:12 +0900
categories: Gen AI Lecture
---

> 들어가기: 본 블로그 글은 S. Ermon, Y. Song, CS236 Deep Generative Models, Stanford University의 강의를 참고하여 제작되었습니다.

### Recap

우리는 앞선 논의에서 [문제 정의 링크] 확률 분포 $$p(x)$$를 아는 것에 대해 이야기를 나누었다. 여기서 가장 중요한 문제는 다음과 같다. 

1. 어떻게 $$p(x)$$를 표현할 것인가?
2. 어떻게 $$p(x)$$학습할 것인가?

다시 살펴보자면, 일반적인 결합확률 분포를 위해 체인룰을 사용하면 다음과 같다. 

$$
p\left(x_{1}, x_{2}, x_{3}, x_{4}\right)=p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid x_{1}, x_{2}\right) p\left(x_{4} \mid x_{1}, x_{2}, x_{3}\right)
$$

이는 매우 일반적인 수식이다. 

베이시안 네트워크를 사용하면 다음과 같다.

$$
p\left(x_{1}, x_{2}, x_{3}, x_{4}\right) \approx p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid \enclose{horizontalstrike}{ x_{1}}, x_{2}\right) p\left(x_{4} \mid x_{1}, \enclose{horizontalstrike}{ x_{2}, x_{3}}\right)
$$

취소선으로 삭제된 몇개의 변수들처럼, 위의 네트워크는 몇가지 조건부 독립성을 가정하고 있다.

여기서 뉴럴 모델은 다음과 같은 특징을 가진다.

$$p\left(x_{1}, x_{2}, x_{3}, x_{4}\right) \approx p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p_{\text {Neural }}\left(x_{3} \mid x_{1}, x_{2}\right) p_{\text {Neural }}\left(x_{4} \mid x_{1}, x_{2}, x_{3}\right)$$

뉴럴 모델은 특정한 함수를 사용하여 조건을 표현한다. 충분히 깊은 뉴럴네트워크가 어떠한 함수도 가정할 수 있다는 장점을 갖고 있다.

### Motivating Example: MNIST

MNIST데이터셋을 살펴보자. 데이터셋 $$\mathcal{D}$$은 다음과 같은 이미지들로 구성되어 있다.

[MNIST이미지]

각 이미지는 $$n=28 \times 28=784$$의 픽셀로 구성되어 있다. 각각의 픽셀은 검은색 (0) 혹은 하얀색 (1)로 나타내어진다.

여기서 결합확률분포 $$p(x)=p\left(x_{1}, \cdots, x_{784}\right)$$를 계산한다. 가능한 경우의 수는 $$x \in\{0,1\}^{784}$$이다. 물론  얻어지는 이미지는 $$x \sim p(x), x$$는 0~9 중 하나의 숫자처럼 보여야 한다.

따라서 우리는 다음과 같은 두가지에 대해 고민해볼 것이다. 

1. 모델의 군집에 대해 표현해보기 $$\left\{p_{\theta}(x), \theta \in \Theta\right\}$$ (지금 주로 다룰 내용)
2. 트레이닝 데이터 $$\mathcal{D}$$를 이용하여 모델의 파라미터 $$\theta$$를 탐색하기 (다음에 다룰 내용)

### Autoregressive Models

MNIST이미지에 대해 임의의 순서를 지정할 수 있다고 해보자. 예를 들면 raster scan (한줄 한줄씩 왼쪽에서 오른쪽으로 픽셀을 스캔)하여서 좌상단의 픽셀을 $$\left(X_{1}\right)$$ 우하단의 픽셀을 $$\left(X_{n=784}\right)$$로 표현한다.

일반성을 일지 않으면서, 체인룰을 활용하여 요소화 (factorization)을 할 수 있다.

$$
p\left(x_{1}, \cdots, x_{784}\right)=p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid x_{1}, x_{2}\right) \cdots p\left(x_{n} \mid x_{1}, \cdots, x_{n-1}\right)
$$

이러한 조건들을 모두 다 표현하기에는 너무 복잡하니 다음과 같이 가정해보자. 

$$p(x_1, \cdot\cdot\cdot, x_{784}) = p_\mathrm{CPT}(x_1;\alpha^1)p_\mathrm{logit}(x_2 | x_1;\boldsymbol{\alpha}^2)p_\mathrm{logit}(x_3 | x_1, x_2;\boldsymbol{\alpha}^3) \cdot\cdot\cdot $$

$$p_{logit}(x_n | x_1, \cdot\cdot\cdot, x_{n-1};\boldsymbol{\alpha}^n)$$

좀 더 명확히 적자면 다음과 같다.

- $$p_{\mathrm{CPT}}\left(X_{1}=1 ; \alpha^{1}\right)=\alpha^{1}, p\left(X_{1}=0\right)=1-\alpha^{1}$$
- $$p_{\mathrm{logit }}\left(X_{2}=1 \mid x_{1} ; \boldsymbol{\alpha}^{2}\right)=\sigma\left(\boldsymbol{\alpha}_{0}^{2}+\boldsymbol{\alpha}_{1}^{2} x_{1}\right)$$
- $$p_{\mathrm{logit }}\left(X_{3}=1 \mid x_{1}, x_{2} ; \boldsymbol{\alpha}^{3}\right)=\sigma\left(\boldsymbol{\alpha}_{0}^{3}+\boldsymbol{\alpha}_{1}^{3} x_{1}+\boldsymbol{\alpha}_{2}^{3} x_{2}\right)$$

이러한 것은 modeling assumption이라 부른다. 여기서는 파라미터로 표현 가능한 함수를 활용하고 있는데 (여기서는 logistic regression) 다음 픽셀이 이전 픽셀들을 이용해서 예측하는 것을 가정하고 있다. 이러한 모델을 **autoregressive models**라고 부른다.

### Fully Visible Sigmoid Belief Network (FVBSN)

위의 수식에서 정리한 과정은 재미있게도 FVBSN과 정확히 동일한 아이디어이다.

우선 조건부 확률은 $$X_{i} \mid X_{1}, \cdots, X_{i-1}$$ 파라미터를 활용한 베르누이 분포를 따른다. 즉,

$$
\hat{x}_{i}=p\left(X_{i}=1 \mid x_{1}, \cdots, x_{i-1} ; \boldsymbol{\alpha}^{i}\right)=p\left(X_{i}=1 \mid x_{<i} ; \boldsymbol{\alpha}^{i}\right)=\sigma(\alpha_{0}^{i}+\sum_{j=1}^{i-1} \alpha_{j}^{i} x_{j})
$$

그렇다면 어떻게 $$p\left(x_{1}, \cdots, x_{784}\right)$$를 정량화(evaluation) 것인가? 이럴 경우 각 조건부 요소들을 모두 곱해주면 된다.

$$p\left(x_{1}, \cdots, x_{784}\right) 
=\left(1-\hat{x}_{1}\right) \times \hat{x}_{2}\left(X_{1}=0\right) \times \hat{x}_{3}\left(X_{1}=0, X_{2}=1\right) \times\left(1-\hat{x}_{4}\left(X_{1}=0, X_{2}=1, X_{3}=1\right)\right)$$

그렇다면 $$p\left(x_{1}, \cdots, x_{784}\right)$$에서 샘플링은 어떻게 할 것인가? (어떻게 이미지를 생성해 낼 것인가?) 이미 정의된 구조를 활용하여 연쇄적으로 샘플링을 수행하면 된다.

1. $$\bar{x}_{1} \sim p\left(x_{1}\right)$$을 이용해 샘플링을 한다. $$\hat{x}_{1}$$을 x1이라 하고, numpy를 쓴다면 다음과 같다.
{% highlight ruby %}
np.random.choice([1,0], p=[x1,1-x1])
{% endhighlight %}
2. $$\bar{x}_{2} \sim p\left(x_{2} \mid x_{1}=\bar{x}_{1}\right)$$를 이용해 샘플링
3. $$\bar{x}_{3} \sim p\left(x_{3} \mid x_{1}=\bar{x}_{1}, x_{2}=\bar{x}_{2}\right) \cdots$$을 이용해 샘플링

연쇄적으로 샘플링을 해야 하기에 속도가 느려질 수 밖에 없다. 그렇다면 이러한 결합분포를 표현하기 위해서는 몇 개의 파라미터가 필요할까?

$$1+2+3+\cdots+n \approx n^{2} / 2$$

앞에서 보았던 $$2^n$$보다 훨씬 적어진 것을 볼 수 있다. 물론 auto regressive한 가정이 들어간 덕이기도 하다.

[결과이미지]

FVBSN의 결과는 어떨까? 왼쪽의 그림은 학습에 사용된 Caltech 101 Silhouettes 데이터셋이다. 물체의 실루엣을 이진 이미지로 나타낸 데이터 셋이다. 오른쪽에는 생성한 (샘플링한) 결과들이 나타나 있다. 
*Figure from Gan et al., Learning Deep Sigmoid Belief Networks with Data Augmentation, 2015. 

### Neural Autoregressive Density Estimation (NADE)

이전 모델의 단점은 무엇일까? 우선 logistic regression에 의존하는 것에서 벗어나서 neural network를 활용해볼 수 있을 것이다. 즉, 하나의 레이어로 구성된 뉴럴네트워크로 바꾼 모델이 NADE이다. 

$$\mathrm{h}_{i}=\sigma\left(A_{i} \mathrm{x}_{<i}+\mathrm{c}_{i}\right)$$

$$
\hat{x}_{i}=p(x_{i} \mid x_{1}, \cdots, x_{i-1} ; \underbrace{A_{i}, \mathrm{c}_{i}, \boldsymbol{\alpha}_{i}, b_{i}}_{\mathrm {parameters }})=\sigma\left(\boldsymbol{\alpha}_{i} \mathrm{~h}_{i}+b_{i}\right)
$$

결국 다음과 같은 hidden variable로 표현이 가능하다. 

$$
\mathrm{h}_{2}=\sigma\left(\underbrace{(\vdots)}_{A_{2}} x_{1}+\underbrace{(\vdots)}_{c_{2}}\right),~~~
\mathrm{h}_{3}=\sigma\left(\underbrace{\left(\vdots\vdots\right)}_{A_{3}}\left( \begin{matrix} x_{1} \\ x_{2} \end{matrix} \right)+\underbrace{(\vdots)}_{c_{3}}\right)
$$

여기서 추가적인 트릭은 **Tie weights**가 있다. 즉, 반복적으로 활용되는 weight들을 반복적으로 활용하도록 'tie'를 하여 계산량을 줄이는 것이다.

$$\mathrm{h}_{i}=\sigma\left(W_{\cdot,<i} \mathrm{x}_{<i}+\mathrm{c}\right)$$

$$\hat{x}_{i}=p\left(x_{i} \mid x_{1}, \cdots, x_{i-1}\right)=\sigma\left(\boldsymbol{\alpha}_{i} \mathrm{~h}_{i}+b_{i}\right)$$

예를 들어서 $$\mathbf{w}_1$$를 $$\mathrm{h}_{2}, \mathrm{h}_{3}, \mathrm{h}_{4}$$를 위해 반복적으로 사용한다고 하면 앞의 수식은 다음과 같이 간략해진다.

$$
\mathrm{h}_{2}=\sigma\left(\underbrace{\left( \begin{matrix} \vdots \\ \mathbf{w}_{1} \\ \vdots \end{matrix} \right)}_{A_{2}} x_{1} \right), ~~
\mathrm{h}_{3}=\sigma\left(\underbrace{\left( \begin{matrix} \vdots & \vdots \\ \mathbf{w}_{1} & \mathbf{w}_2 \\ \vdots & \vdots \end{matrix} \right)}_{A_{3}} \left( \begin{matrix} x_{1} \\ x_{2} \end{matrix} \right) \right), ~~
\mathrm{h}_4
=\sigma\left(\underbrace{\left( \begin{matrix} \vdots & \vdots & \vdots \\ \mathbf{w}_{1} & \mathbf{w}_2 & \mathbf{w}_3 \\ \vdots & \vdots & \vdots \end{matrix} \right)}_{A_{3}} \left( \begin{matrix} x_{1} \\ x_{2} \\ x_{3} \end{matrix} \right) \right)
$$

만약 $$\mathrm{h}_{i} \in \mathbb{R}^{d}$$이라면, 얼마나 많은 파라미터가 필요할까?

답은 $$n$$에 대해 선형적이다. 가중치 값들 $$W \in \mathbb{R}^{d \times n}$$, 바이어스 $$c \in \mathbb{R}^{d}$$ 그리고 $$n$$개의 logistic regression coefficient vectors
$$\boldsymbol{\alpha}_{i}, b_{i} \in \mathbb{R}^{d+1}$$를 고려하면 된다. 따라서 확률은 $$O(n d)$$의 계산 복잡도로 평가가 가능하다.

[NADE결과]

NADE의 결과는 어떨까? 왼쪽에는 생성한 샘플들이, 그리고 오른쪽에는 $$\hat{x_i}$$의 확률들이 나타나 있다. 왼쪽과 오른쪽의 차이에 대해 종종 질문을 받는다. 오른쪽은 0 혹은 1로 샘플링을 하기 위한 베르누이 분포를 의미한다. 왼쪽은 이러한 분포로 실제 0 혹은 1로 샘플링 된 결과를 의미한다.
*Figure from Larochelle et al., The Neural Autoregressive Distribution Estimator, 2011.

### General discrete distributions

이제 이 아이디어를 0과 1이 아닌 일반적인 변수들에 대해서 어떻게 표현할 수 있을까? $$X_{i} \in\{1, \cdots, K\}$$ 예를 들어 픽셀의 값들이 0에서 255까지 있는 일반적인 상황을 가정할 수 있다. $$K=256$$

한가지 방법은 $$\hat{\boldsymbol{x}}_{i}$$를 categorical distribution으로 표현하는 것이다.

$$\mathrm{h}_{i}=\sigma\left(W_., _{<i}\mathrm{x}_{<i}+\mathrm{c}\right)$$

$$p\left(x_{i} \mid x_{1}, \cdots, x_{i-1}\right)=\operatorname{Cat}\left(p_{i}^{1}, \cdots, p_{i}^{K}\right)$$

$$\hat{\boldsymbol{x}}_{i}=\left(p_{i}^{1}, \cdots, p_{i}^{K}\right)=\operatorname{softmax}\left(A_{i} \mathrm{~h}_{i}+\mathrm{b}_{i}\right)
$$

따라서 softmax가 sigmoid/logistic function인 $$\sigma(\cdot)$$를 일반화하고, $$K$$ 개의 벡터 $$a$$를 $$K$$의 확률을 가진 벡터 한개로 변형한다는 것을 알 수 있다. (양의 실수로서 합이 1이다.)

$$
\operatorname{softmax}(a)=\operatorname{softmax}\left(a^{1}, \cdots, a^{K}\right)=\left(\frac{\exp \left(a^{1}\right)}{\sum_{i} \exp \left(a^{i}\right)}, \cdots, \frac{\exp \left(a^{K}\right)}{\sum_{i} \exp \left(a^{i}\right)}\right)
$$

numpy를 활용한다고 하면, 다음과 같다.

{% highlight ruby %}
np.exp(a)/np.sum(np.exp(a)) 
{% endhighlight %}

### RNADE

그렇다면 연속적인 랜덤 변수 $$X_{i} \in \mathbb{R}$$는 어떻게 표현할 것인가? 예를 들면, speech signal이 있을 것이다. 해결책은 $$\hat{\boldsymbol{x}}_{i}$$를 연속적인 분포로 나타내면 된다. 예를 들어 선형적으로 $$K$$개의 가우시안을 표현하면 다음과 같다.

$$
p\left(x_{i} \mid x_{1}, \cdots, x_{i-1}\right)=\frac{1}{K}  \sum_{j=1}^{K} \mathcal{N}\left(x_{i} ; \mu_{i}^{j}, \sigma_{i}^{j}\right)
$$

$$
\mathrm{h}_{i}=\sigma\left(W_{\cdot,<i} \mathrm{x}_{<i}+\mathrm{c}\right)
$$

$$
\hat{\boldsymbol{x}}_{i}=\left(\mu_{i}^{1}, \cdots, \mu_{i}^{K}, \sigma_{i}^{1}, \cdots, \sigma_{i}^{K}\right)=f\left(\mathrm{h}_{i}\right)
$$

여기서 $$\hat{\boldsymbol{x}}_{i}$$는 각 가우시안의 평균과 표준편차 $$(\mu_{i}^{j}, \sigma_{i}^{j})$$을 의미한다. 여기서 exponential함수 $$\exp (\cdot)$$를 활용해 표준편차가 양의 값을 가지도록 한다.