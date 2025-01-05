---
layout: post
title:  "GenAI Lecture: 02 Representation"
date:   2025-01-05 01:09:12 +0900
categories: Gen AI Lecture
---

> 들어가기: 본 블로그 글은 S. Ermon, Y. Song, CS236 Deep Generative Models, Stanford University의 강의를 참고하여 제작되었습니다.

### 컬러이미지의 생성?

이제 앞선 글에 이어서 이미지 생성을 위해 어떤 표현법을 써야 할 지 생각해보자. 이미지는 하나의 화소 혹은 픽셀 (pixel)로 표현이 된다. 흔히 말하는 HD퀄리티의 이미지는 가로로 1280개 세로로 720개 픽셀로 구성이 되며, 총 92만여개의 픽셀로 표현이 된다. 각 픽셀은 모니터가 표현 가능한 색상으로 표현되기 위해, 빛의 3원색(Red, Green, Blue)으로 표현이 되고, 각 색은 8bit의 값을 활용해 총 $$2^8=256$$가지 밝기의 색을 각각 표현할 수 있다. 

앞서 공부해본 categorical distribution으로 위의 경우의 수를 표현해보면, 다음과 같이 정리 될 수 있다.

- Red channel: $$Val(R) = \{0, \cdots, 255\}$$
- Green channel: $$Val(G) = \{0, \cdots, 255\}$$
- Blue channel: $$Val(B) = \{0, \cdots, 255\}$$

여기서 $$R$$, $$G$$, 그리고 $$B$$는 랜덤 변수 (random variable)을 의미한다. 그렇다면, 동시확률분포 (joint distribution)을 활용하여 픽셀 값 하나를 샘플링 한다고 가정해보자. 그렇다면 $$(r, g, b) \sim p(R, G, B)$$이 될 것이다. 

앞선 Bernoulli distribution을 활용하여 특정한 값을 표현하려면, $$p(R=r, G=g, B=b)$$, 무려 아래와 같은 경우의 수를 감안해야 한다.

$$256 \times 256 \times 256 -1$$

이미지 한장도 아니고, 픽셀 하나만을 위한 경우의 수인데, 지나치게 많지 않은가? HD퀄리티의 영상 하나를 만들기 위해 92만개의 픽셀을 이런식으로 만든다고 가정하면, 감안해야 할 경우의 수가 너무나도 많다. 실제로 앞의 biased dice예시에서는 $$m-1$$개의 변수만이 필요하였으나, 이미지 생성을 위한 변수는 기하 급수적으로 늘어남을 알 수 있다.

$$(256^3-1)\times(1280\times720)\sim1.5462^{13}$$

### 이진이미지의 생성?

아직 컬러 이미지를 생성하기에는 좀 이른 것으로 보인다. 그러면, 더 문제를 간략하게 생각하여 이진 이미지 생성을 생각해보자. $$n$$개의 픽셀이 있을때, $$X_1,...,X_n$$, $$Val(X_i)=\{0,1\} =\{\text{Black, White} \}$$ 이진 이미지란 각 픽셀이 0 또는 1만 가지고 있는 흑과 백만 있는 영상이다. 이렇게 하면 경우의 수를 많이 줄일 수 있을까?

$$\underbrace{2\times2\times \cdots \times 2}_{\text{n times}} = 2^n$$

$$p(x_1, \cdots, x_n)$$를 이용해 샘플링을 하면 이미지를 얻게 되지만, 이러한 결합확률분포 $$p(x_1, \cdots, x_n)$$를 $$n$$개의 픽셀에 대해 수행하려면 여전히 다음과 같은 경우의 수가 필요하다.

$$2^n-1$$

결합확률분포는 각각 구성하는 랜덤변수가 독립일때 분리하여 표현할 수 있다. 이를 테면, 

$$
p\left(x_{1}, \ldots, x_{n}\right)=p\left(x_{1}\right) p\left(x_{2}\right) \cdots p\left(x_{n}\right)
$$

위의 경우에도 가능한 경우의 숫자는 무엇인가? $$ 2^{n}$$ 

잠깐, 이미지를 생성하려고 하는데 각 픽셀을 독립적으로 표현하는게 정말 도움이 될까?

[weird binary image예시들]

### 핵심 수식

확률 분포를 바꾸어 생각해보자. 그 이전에 앞서서, 아이디어 전환의 핵심이 될 수식 두가지를 소개한다. 

**체인 룰(Chain rule)**: 만약 $$S_{1}, \ldots S_{n}$$이 사건(event)라 하고, 모두 발생할 확률이 있다고 하면 $$p\left(S_{i}\right)>0$$ 다음과 같은 수식이 만족한다.

$$
p\left(S_{1} \cap S_{2} \cap \cdots \cap S_{n}\right)=p\left(S_{1}\right) p\left(S_{2} \mid S_{1}\right) \cdots p\left(S_{n} \mid S_{1} \cap \ldots \cap S_{n-1}\right)
$$

**베이스 룰(Bayes' rule)**:
또한, 두가지 발생할 가능성이 0이 아닌 사건에 대해, 즉, $$S_{1}, S_{2}$$ $$p\left(S_{1}\right)>0$$ 이고 $$p\left(S_{2}\right)>0$$이라면, 다음이 성립한다.

$$
p\left(S_{1} \mid S_{2}\right)=\frac{p\left(S_{1} \cap S_{2}\right)}{p\left(S_{2}\right)}=\frac{p\left(S_{2} \mid S_{1}\right) p\left(S_{1}\right)}{p\left(S_{2}\right)}
$$

이제 다시 한번 이진 이미지 생성의 예시로 돌아와 보자. 원래의 수식에서 $$p\left(x_{1}, \ldots, x_{n}\right)$$ 체인룰을 적용하면, 다음과 같이 변환하는 것이 가능하다.

$$
p\left(x_{1}, \ldots, x_{n}\right)=p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid x_{1}, x_{2}\right) \cdots p\left(x_{n} \mid x_{1}, \cdots, x_{n-1}\right)
$$

위의 체인룰은 결합확률분포를 연쇄적인 결합확률 분포로 구조를 바꾸는 것을 제외하고는, 원래 수식의 의미를 변환하지 않는 다는 점을 유의하자.

이제 체인 룰을 쓴 상황에서 파라미터가 몇개 필요한지 다시 생각해보자.

- $$p\left(x_{1}\right)$$는 1개의 파라미터면 된다.
- $$p\left(x_{2} \mid x_{1}\right)$$는 몇개의 파라미터가 필요할까?
    - $$x_{1}$$에 대한 경우의 수를 나누어 생각하면 된다. 즉, 다음과 같다.
    - $$p\left(x_{2} \mid x_{1}=0\right)$$는 1개의 파라미터를 필요로 한다. 
    - $$p\left(x_{2} \mid x_{1}=1\right)$$는 1개의 파라미터를 필요로 한다.
    - 즉 2개의 파라미터가 필요하다.
그렇다면 전체 파라미터의 숫자는 몇개일까?

$$1+2+\cdots+2^{n-1}=2^{n}-1$$

체인룰은 아쉽게도 파라미터 개수를 줄이는데에는 도움을 주지 못한다. 

### 몇가지 트릭

그렇다면 다음과 같은 변화를 주는 건 어떨까? 

$$X_{i+1} \perp X_{1}, \ldots, X_{i-1} \mid X_{i}$$

쉽게 말해 $$i+1$$번째 픽셀 $$X_{i+1}$$은 $$X_{i}$$에만 의존하고, 다른 기존 픽셀들 $$X_{1}, \ldots, X_{i-1}$$과는 의존 관계가 없다. 그렇게 된다면, 다음과 같이 수식을 바꿀 수 있다.

$$
\begin{aligned}
p\left(x_{1}, \ldots, x_{n}\right) & =p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid x_{1}, x_{2}\right) \cdots p\left(x_{n} \mid x_{1}, \ldots, x_{n-1}\right) \\
& =p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid x_{2}\right) \cdots p\left(x_{n} \mid x_{n-1}\right)
\end{aligned}
$$

이렇게 되면 파라미터가 몇 개일까? 

$$2 n-1$$

Exponential reduction이다!