---
layout: post
title:  "GenAI Lecture: 02 Representation"
date:   2025-01-05 01:09:12 +0900
categories: Gen AI Lecture
---

> 들어가기: 본 블로그 글은 S. Ermon, Y. Song, CS236 Deep Generative Models, Stanford University의 강의를 참고하여 제작되었습니다.

### Bayesian Network의 응용 - 스팸 메일 탐지

Bayesian Network를 이용한 간단한 응용을 생각해보자. 이메일이 스팸 이메일인지 아닌지를 구분하여 보는 것이다. 우선 레이블을 지정해야 하는데, $$(Y=1)$$을 스팸 이메일로 $$(Y=0)$$을 스팸이 아닌 메일로 정의해보자.

- 영어 단어에 등장하는 단어를 $$[1: n]$$로 번호를 매긴다고 가정하자.
- $$X_{i}=1$$를 워드 $$i$$가 이메일에서 등장하면 1로 두고 그렇지 않으면 0으로 둔다. 
- 결과적으로 이메일은 특정한 분포를 따라 나타나게 된다. $$p\left(Y, X_{1}, \ldots, X_{n}\right)$$

그렇다면 $$p\left(Y, X_{1}, \ldots, X_{n}\right)$$를 어떻게 표현하여야 할까? 한가지 해볼 수 있는 접근은 단어들이 주어진 변수 $$Y$$에 대해 조건부 독립으로 두고 아래와 같이 두는 방법이 있다.

$$
p\left(y, x_{1}, \ldots x_{n}\right)=p(y) \prod_{i=1}^{n} p\left(x_{i} \mid y\right)
$$

이러한 구성은 우리가 임의로 베이시안 네트워크를 구성한것과 같다. 스팸이메일, 스팸이 아닌 이메일로  이미 레이블링이 된 트레이닝 데이터가 있다면, 다음과 같이 베이시안 룰을 적용하여 수식을 정리할 수 있다. 

$$
p\left(Y=1 \mid x_{1}, \ldots x_{n}\right)=\frac{p(Y=1) \prod_{i=1}^{n} p\left(x_{i} \mid Y=1\right)}{\sum_{y=\{0,1\}} p(Y=y) \prod_{i=1}^{n} p\left(x_{i} \mid Y=y\right)}
$$

다만 한가지 의문이 든다. 여기 있는 변수간의 독립성의 가정이 합리적일까? 당연히 그렇지 않을 것이다. 한가지 사실은 대부분의 이러한 가정이 잘못되었지만, 대부분은 그럼에도 유용하다는 사실이다.


### 결정 모델과 생성 모델의 차이?

좀 더 쉬운 예시를 생각해보자. 체인룰을 이용하면, 다음과 같은 두가지 방식으로 수식을 전개 할 수 있다. 

$$p(Y, X)=p(X \mid Y) p(Y)=p(Y \mid X) p(X)$$

체인룰로 어떤 변수부터 풀어서 쓸 것인지의 차이일 뿐, 의미적으로는 같다. 그리고, 각각에 대한 베이시안 네트워크는 그림과 같다. 어떤것이 좋을까?

하지만 우리가 앞선 task를 위해서 필요한 것은 $$p(Y \mid X)$$이다. 이 관점으로 다시 생각해보면, 왼쪽 모델에서 필요한 것은 $$p(Y)$$와 $$p(X \mid Y)$$를 얻거나 학습한 다음, $$p(Y \mid \mathrm{X})$$를 계산하는 방법이다.

오른쪽 모델에서는 단순히 조건부 확률 $$p(Y \mid \mathrm{X})$$만 계산하는 것으로 충분하다. 즉 $$p(\mathrm{X})$$를 학습하거나 배우거나 활용할 필요가 없음을 의미한다. 이러한 모델은 주어진 $$X$$에 대해서 $$Y$$를 결정짓는데에만 집중하기 때문에 discriminative모델이라 부른다. 

### 스팸 메일 탐지 문제로 돌아와서

앞선 스팸 메일을 찾는 문제로 돌아와보자. 그러면 같은 방법으로 체인룰을 이용해 다음과 같이 표현할 수 있다.

Generative approach:

$$p(Y, X)=p(Y) p\left(X_{1} \mid Y\right) p\left(X_{2} \mid Y, X_{1}\right) \cdots p\left(X_{n} \mid Y, X_{1}, \cdots, X_{n-1}\right)$$

Discriminative approach:

$$p(Y, X)=p\left(X_{1}\right) p\left(X_{2} \mid X_{1}\right) p\left(X_{3} \mid X_{1}, X_{2}\right) \cdots p\left(Y \mid X_{1}, \cdots, X_{n-1}, X_{n}\right)$$

생각을 해보면 다음과 같은 선택을 해야 함을 알 수 있다.

- **[선택 #1]** 생성 모델에서 $$p(Y)$$를 구하는 것은 간단하다. 하지만 $$p\left(X_{i} \mid X_{p a(i)}, Y\right)$$를 파라미터화 하는것이 의문이다.
- **[선택 #2]** 구분 모델에서는 $$p(Y \mid X)$$를 파라미터화 하는 것이 곤란하다. 여기서 우리는 $$p(X)$$에 대해서는 고민할 필요가 없다. 왜냐하면 $$X$$가 언제나 classiciation 문제에서는 주어지기 때문이다. 

**[선택 #1]**과 관련해서는 우리는 $$X_{i} \perp \mathrm{X}_{-i} \mid Y$$을 가정할 수 있다. 간단하다.

### Logistric Regression

**[선택 #2]**와 관련해서는 다음을 가정하자: $$
p(Y=1 \mid \mathrm{x} ; \boldsymbol{\alpha})=f(\mathrm{x}, \boldsymbol{\alpha})
$$ 이제부터는 베이시안 네트워크를 활용하기 위해 테이블로 표현하지 않고, $$x$$를 활용하기 위한 함수로 표현을 하고 회귀문제 (regression)문제로 변환을 하게 된다. 

- 값을 0과 1로 둔다.
- 간단하지만 합리적인 방법으로 $$x_{1}, \cdots, x_{n}$$를 표현한다.
- 하나의 벡터 $$\alpha$$를 이용해 $$n+1$$를 파라미터를 활용한다.

이러한 결과를 위해 선형 의존성을 활용할 수 있다. $$z(\boldsymbol{\alpha}, \mathrm{x})=\alpha_{0}+\sum_{i=1}^{n} \alpha_{i} x_{i}$$로 두면, $$p(Y=1 \mid \mathrm{x} ;\boldsymbol{\alpha})=\sigma(z(\boldsymbol{\alpha}, \mathrm{x}))$$가 되고, $$\sigma(z)=1 /\left(1+e^{-z}\right)$$는 logistic function으로 표현이 된다.

Logistic regression의 특징은 다음과 같다.
1. $$p(Y=1 \mid \mathrm{x} ; \boldsymbol{\alpha})>0.5$$이 $$\mathrm{x}$$에 대해 선형이고,
2. 동일한 확률 contour는 선형 라인이다.
3. 확률 비율의 차이는 매우 구체적이다. 

Logistic model은 앞의 generative approach에서 사용했던 navie Bayes기반 방식과 달리 $$X_{i} \perp X_{-i} \mid Y$$를 가정하지 않는다. 이러한 차이는 다양한 응용에서 큰 변화를 가져올 수 있다. 

예를 들어 스팸 이메일 구분 문제에서 $$X_{1}=1$$ ("bank"라는 단어가 이메일에 있음) $$X_{2}=1$$ ("account"라는 단어가 이메일에 있음). 이러한 경우 스팸인지 아닌지에 무관하게 이러한 두개의 단어는 항상 비슷하게 등장하게 된다. 즉, $$X_{1}=X_{2}$$ 하지만, 이러한 경우 naive Bayes를 활용하게 되면 $$p\left(X_{1} \mid Y\right)=p\left(X_{2} \mid Y\right)$$가 되어 두번 카운팅 하는게 된다. 반면에 logistic regression을 활용하면 $$\alpha_{1}=0$$ 혹은 $$\alpha_{2}=0$$를 하여 효과적으로 무시를 할 수 있게 된다. 

그렇다면, Generative model이 유용하지 않은 걸까? 그렇지 않다. 조건부 모델을 사용하려면 항상 $$X$$가 관측 가능하다는 조건이 필요한데, 만약 $$X_i$$가 없다면, 생성 모델이 여전히 보이지 않았던 단어까지 주변화(maganilzation)하여 $$p\left(Y \mid X_{\text {evidence }}\right)$$를 계산하게 해준다.

### Neural Models

다시 스팸 메일 구분 문제에서 결정 모델을 살펴보면, 다음과 같은 가정을 하였다. 

$$
p(Y=1 \mid \mathrm{x} ; \boldsymbol{\alpha})=f(\mathrm{x}, \boldsymbol{\alpha})
$$

여기서 logistic function을 활용해 선형 의존성 (linear dependence)를 만들었다. 즉 

$$z(\boldsymbol{\alpha}, \mathrm{x})=\alpha_{0}+\sum_{i=1}^{n} \alpha_{i} x_{i}$$
$$p(Y=1 \mid \mathrm{x} ; \boldsymbol{\alpha})=\sigma(z(\boldsymbol{\alpha}, \mathrm{x}))$$, $$\sigma(z)=1 /\left(1+e^{-z}\right)$$ 로 두었으나 선형 의존성이 너무 단순하다.

이제 비선형 의존성을 추가하기 위해 다음과 같은 변형함수를 사용해보면 다음과 같다. 

$$h(A, b, x)=f(A x+b)$$

$$p_{\text {Neural }}(Y=1 \mid \mathrm{x} ; \boldsymbol{\alpha}, A, \mathrm{~b})=\sigma\left(\alpha_{0}+\sum_{i=1}^{h} \alpha_{i} h_{i}\right)\
{}$$

이러한 neural network를 사용한 구조는 다음과 같은 장점이 있다.
- 좀 더 유연성을 가진다.
- 더 많은 파라미터를 가진다. $$A, \mathrm{b}, \boldsymbol{\alpha}$$
- 여러번의 반복을 거쳐야 한다. 

### Bayesian Network과 Neural Network의 차이?

이제 지금까지 논의한 내용을 정리해보겠다. 일반적인 결합확률 분포를 위해 체인룰을 사용하면 다음과 같다. 

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