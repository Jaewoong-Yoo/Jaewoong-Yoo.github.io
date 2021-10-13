---
layout: single   
title: "RL Lecture 7. Policy Gradient Methods"   

---

# Policy Gradient Methods

## Intro.


Value-Based vs. Policy-Based RL
- Value Based
    - Learnt Value Function
    - Implicit Policy (e.g. $\epsilon$-greedy)
- Policy Based
    - No Value Function
    - Learnt Policy
- Actor-Critic
    - Learnt Value Function
    - Learnt Policy

$ $

Advantages of Policy-Based RL   
이점   
- 수렴성 good
- continuous action에서는 action을 전부 넣어보고 q value function을 학습해야한다? q하나 학습시키는데 모든 액션 넣어보고 또 optimization 과정을 거친다는게 현실적으로 어려움. (풀릴지도 사실 잘 모름)
- stochastic policy 학습 가능
- stable 함 (theta를 조금씩 update하므로)

단점   
- variance가 크고 비효율적임 (60% -> 62% 이런식이라 stable하지만 효율 떨어짐)
- 보통 global 보다 local로 수렴하게 됨

### Policy Search

#### Policy Objective Functions

- Policy 목적함수를 정의하는 세가지 방법론
1. episodic environments (한 판씩 끝나는 환경) (start value 사용 가능한) 에서
2. continuing environments (각 state에서 있을 확률 x 그 state에서의 value의 총합) (이걸 average value라 함)
3. average reward per time-step = $\sum$($\sum$(각 state에서 이 policy로 action 한 번 하고 그 때 얻는 reward X $\pi$의 확률 가중치) X stationary distribution $d^{\pi_\theta}$ 가중치)

하지만 위의 세가지 방법에 대해서 똑같이 작동하는 것만 배울 것임 

## Finite Difference Policy Gradient



- 여기서 policy gradient는 J에 대해서 gradient를 update하는거지 $\pi$에 대해서 없데이트 하는 것 아님! (단지 $\pi$가 쓰일 수 있는 것뿐)

- Finite Difference Policy Gradient를 구하는 방법론이 제일 무식하고 쉬운 방법론 임
- evaluation을 n번 반복하면서 미분하고 update해야하는데... 그게 전부하려면 어려움
- 단순하지만 noisy하고 비효율적임
- 하지만 또 다른 장점은 policy가 미분가능하지 않아도 업데이트 가능


## Monte-Carlo Policy Gradient

- **가정: policy가 differentialbe하고 $\pi$에 대한 gradient를 안다.**
- 여기서 핵심은 **Likelihood ratios** 트릭은 사용했다는걸 이해하는 게 핵심!! (계산을 편하게 하기 위해)
- Likelihood ratios : logx미분 = 1/x




### one-step MDPs 에서 보면...

- 기대값 : 어느 state에 분호할지에 대한 확률 X 그 state에서 어떤 action할지의  확률
- 그래서 결국 $\nabla_\theta J(\theta)$ 가 Expectation의 term으로 표현이 되었다는 것이 중요!
- 예를 들면, 주사위를 던져서 policy  $\pi$를 따라 계속 행동을 한 다음에, 각 행동 (s,a)에 대해서 $\nabla_\theta log$~ 를 구하고 r을 곱해준 값이 gradient J가 되는 것! --> 이 성립을 위해 **likelihood ratio trick**이 사용된 것

### Policy Gradient Theorem
multi-step MDPs에서 보면...

앞 one-step MDPs에서의 likelihood ration approach를 일반화 하기위해서,   
**Policy Gradient Theorem**을 사용!!!   
- one-step MDPs의 r 자리에 Q(그 step에서 action을 했을 때 얼마를 받을 지의 r총합)를 넣으면 똑 같음!
- 앞에서 말한 목적함수 정의 방법로 3가지를 다 사용해도 된다는 설명이 Policy Gradient Theorem


#### MC Policy Gradient (REINFORCE)


- 우린 신이 아니므로 앞의 Q의 값을 정화히 모름. 그래서 return을 쓴다.
- retrun(accumulated discountive reward)은 Q의 undiased sample 이므로.. -> game끝까지 해보고 얻은 return을 value 학습 때 쓰는 것과 동일

REINFORCE pseudocode code보면... (알파고에서 쓰임)
- value function 쓰지 않고 policy를 바로 update하므로 Policy-Based 방법론임
- Q 자리에 return(v)를 쓰고 있으므로 -> Monte-Carlo 방법론
- 그래서  Montel-Carlo Policy Gradient 방법론이다.

## Actor-Critic Policy Gradient

- Policy Gradient Theorem을 생각해보자, gradient log $\pi$(score) X Q 의 기댓값이 gradient J잖아.
- REINFORCE는 Q자리에 return을 넣었지만, 그럴필요 없이 그냥 Q를 따로 학습해서 넣어주면 되지 않을까? -> 된다!! 이게 Actor-Critic
- Q : 어떤 $\pi$를 따라서 게임을 끝까지 했을 때의 기댓값
- Q는 $\pi$에 종속적인 친구였는데, $\pi$가 계속 바뀌니까 -> Q를 학습해서 Q로 pi를 업데이트하고 pi가 업데이트 되니까 다시 또 Q를 업데이트!  
(마치 policy iteration와 관련 되어 있어 보임)

- 즉, a를 바꾸는 방향으로 Q가 좋으면 더 그 방향으로 더 하게 바꾸고, 안좋으면 덜 하게 바꾸는 것 update!

#### Reducing Variance Using a Baseline

Advantage function -> A = Q - V
ex) 100만워 99만9천의 상대적인 차이가 별로 없으니까 수렴하는데 낭비하게 되는 문제가 있음. 따라서 baseline을 만들어서 -1~1사이의 크기가 되도록 상대적인 크기를 줄인다. 
- state s에 있는 value가 있고, 거기서 action a를 추가적으로 했을 때 value의 차이 가 advantage이다. 이게 바로 baseline
- 따라서 기존 Q자리에 advantage A(=Q-V)가 들어가도 식이 성립! (V에 대한 policy gradient는 0이 되어 바뀌지 않음을 확인했으므로)
- policy graident의 variance를 엄청나게 줄이는 효과있음!

#### Estimating the Advantage Function

- 결론부터 말하자면 **Q가 이제 안 필요함!**
- **TD error가 advantage A의 sample이라고 볼 수 있다.**

수식으로 보면...
- TD error의 기댓값이 advantage A이기 때문에, delta(TD error)의 sample들은 즉, A의 unbiased sample이라는 것!
- delta 하나가 A랑 같지는 않겠지만, 계속해서 평균을 내다보면 A랑 같아진다.

따라서,,,
- A자리에 A의 smaple인 delta(TD error)을 넣을 수 있다.
- 하지만 위는 true value function에서의 이야기이고, 우리의 실전 문제에서는
- true value function을 모방한 V가 있어서 그걸 그냥 쓰면 됨
- 결론적으로, Q를 학습할 필요가 없다는 점을 기억!
- Advantage 자리에 TD error 쓰면 된다.
