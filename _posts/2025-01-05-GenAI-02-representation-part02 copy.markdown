---
layout: post
title:  "GenAI Lecture: 02 Representation"
date:   2025-01-05 01:09:12 +0900
categories: Gen AI Lecture
---

> 들어가기: 본 블로그 글은 S. Ermon, Y. Song, CS236 Deep Generative Models, Stanford University의 강의를 참고하여 제작되었습니다.

앞선 글에서 결합확률함수의 파라미터의 개수를 줄이는 것은 극적으로 가능하다는 것을 보였다.

$$
\begin{aligned}
p\left(x_{1}, \ldots, x_{n}\right) & =p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid x_{1}, x_{2}\right) \cdots p\left(x_{n} \mid x_{1}, \ldots, x_{n-1}\right) \\
& =p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid x_{2}\right) \cdots p\left(x_{n} \mid x_{n-1}\right)
\end{aligned}
$$

이렇게 되면 파라미터가 $$2^n$$에서 $$2 n-1$$개가 된다.

### Bayesian Network

이러한 가정은 각 픽셀이 직전 픽셀에만 영향을 받는다는 가정을 통해 얻어진다. 이러한 가정을 일반적으로 나타내는 기법은 베이시안 네트워크 (Bayesian networks)라 부른다. 

베이시안 네트워크는 방향성을 가진 사이클이 없는 그래프로 표현이 되는데 (Directed Acyclic graph: DAG), 그래프의 특징 상 vertex와 edge로 표현이 된다. $$G=(V, E)$$ 그리고 다음과 같은 특징을 가진다.
- 하나의 노드는 $$i \in V$$ 랜덤 변수 하나를 의미한다. $$X_{i}$$
- 노드 당 하나의 조건부 확률 분포 (conditional probability distribution: CPD) $$p\left(x_{i} \mid \mathrm{x}_{\mathrm{Pa}(i)}\right)$$
를 갖게 되고, 이것은 부모의 값에 따라 조건부 확률을 나타내게 한다.
- 그래프는 $$G=(V, E)$$ 베이시안 네트워크의 구조체라고 부른다.
- 다음과 같은 결합확률분포를 따른다.
$$
p\left(x_{1}, \ldots, x_{n}\right)=\prod_{i \in V} p\left(x_{i} \mid \mathrm{x}_{\operatorname{Pa}(i)}\right)
$$
- 재미있게도 $$p\left(x_{1}, \ldots x_{n}\right)$$가 DAG이면, 유효한 확률 분포이다.

위의 베이시안 네트워크는 결합확률분포를 매우 경제적으로 만들어준다. 왜냐하면, 각 노드 $$i$$에 대해서 조건부 확률에 영향을 주는 부모노드의 개수만 $$\|\mathrm{Pa}(i)\|$$ 파라미터의 개수에 영향을 주기 때문이다. 이는 일반적인 결합확률분포가 $$p\left(x_{1}, \ldots, x_{n}\right)$$를 표현하기 위해 서로 다른 노드를 표현하기 위해 $$\|V\|$$개의 노드를 모두 고려했던 것과 크게 차이가 있다. 따라서 exponential이 $$\|\mathrm{Pa}(i)\|$$에 있게 된다.

[DAG의 그림]

### Bayesian Network 예시

[DAG의 예시]

결합확률분포는 어떻게 표현될까?

$$
\begin{aligned}
p\left(x_{1}, \ldots, x_{n}\right) & =\prod_{i \in V} p\left(x_{i} \mid \mathrm{x}_{\operatorname{Pa}(i)}\right) \\
p(d, i, g, s, l) & =p(d) p(i) p(g \mid i, d) p(s \mid i) p(l \mid g)
\end{aligned}
$$

위의 결합확률분포는 일반적인 체인룰과 다음과 같은 차이가 있다.

$$
p(d, i, g, s, l)=p(d) p(i \mid d) p(g \mid i, d) p(s \mid i, d, g) p(l \mid g, d, i, s)
$$

따라서 다음과 같은 추가적인 랜덤 변수간 독립성이 추가되었음을 알 수 있다.

$$
D \perp I, \quad\quad
S \perp\{D, G\} \mid I, \quad\quad
L \perp\{I, D, S\} \mid G. 
$$

이렇듯, 베이시안 네트워크에서는 CPD를 곱하는 것으로 확률에 따른 경우의 수를 표현할 수 있다. 또한 DAG의 구조에 나타나 있는 방식에 따라 샘플링을 수행할 수도 있다. 이 과정에서 조건부 독립성을 그림을 통해 쉽게 파악할 수 있다. 