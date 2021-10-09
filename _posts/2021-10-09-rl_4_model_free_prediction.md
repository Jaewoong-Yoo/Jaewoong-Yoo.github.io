----
layout: single
title: "RL Lecture 4. Model-Free Prediction"
----

# Model-Free Prediction

- 환경을 모를 때 (transition이나 reward 등 관련) model free prediction 문제를 푸는 방법을 배움
- Prediction = Policy Evaluation (value function을 구하는)
    - 여기서 value function $V(S_t)$은 current state t에서 앞으로 reward를 얼마나 받을 지 예측
- MC vs. TD
$  $

## Monte-Carlo Learning

- episode 끝날 때까지 얻은 return들에 대한 평균을 취하는 방법
- MC는 value function을 actual return **G**의 방향으로 update

## Temporal-Difference Learning

- 한 스텝 앞에 더 나아가서 예측한 value function을 target(=**Temporal-Difference target**)으로 삼고, 그 방향으로 update
- TD(1), TD(2), ..., TD(n) 및 **TD($\lambda$)**의 forward/**backward** view 존재

## Advantages and Disadvantages of MC vs. TD

**vs. - (1)**
- TD는 final outcome이 나오기 전에 학습 할 수 있는 반면, MC는 모든 step이 끝날 때까지 기다린 후 계산된 return 값으로 학습 가능
- 따라서 TD는 continuing(non-terminating) 환경에서도 학습 가능, MC는 terminating 환경에서만 학습 가능


**vs. - (2)**
- MC는 variance가 크고, bias가 zero
    - $V_\pi(S)$ 에 대해 수렴성 good (NN 쓰더라도)
    - 초기 값에 민감하지 않음 (실제 값으로 update하니까)
    - 이해하기 쉽고 사용하기 쉬움

- TD는 variance가 작고, bias가 약간 있음
    - MC보다 조금 효과적
    - TD(0)는 $V_\pi(S)$에 수렴함
    - 처음 추측치가 잘 못되면 수렴 제대로 안해서 초기 추측치에 민감
    $ $

**vs. - (3)**
- TD는 Markov property를 사용해서 value를 추측
    - Markov 환경에 좀 더 효과적
- MC는 Markov property를 사용하지 않음
    - Markov다 버리고, mean squared error를 최소화시킴. non-Markov 환경에 효과적

## Bias/Variance Trade-Off 관점에서 MC vs. TD

- $V_\pi(S_{t+1})$ 의 실제 값을 안다고 하면, $R_{t+1} + V_\pi(S_{t+1})$ 가 True TD target이 되므로 unviased 됨

- 그러나 신이 아닌 이상 $V_\pi(S_{t+1})$ 의 실제 값을 모르기 때문에 위의 TD target은 결국 biased estimate value이다.

- TD target은 return보다 variance가 더 적다. return은 에피소드 끝날 때 까지 무수히 많은  random성과 state transition probability가 존재하므로


## Bootstrapping and Sampling 관점에서 MC vs. TD

- Bootstrapping은 추측치로 추측치를 업데이트 하는 경우 (depth 관점)
    - MC는 끝까지 가고, DP와 TD는 한 스텝만 가니까
        - MC --> bootstrap X  --> deep backups
        - DP --> bootstrap O  --> shallow backups
        - TD --> bootstrap O  --> shallow backups
- Sampling은 하나대해서 샘플을 업데이트 (width 관점)
    - MC와 TD는 했던 것을 가지고 sampel하지만, DP는 모든 가능한 action을 다 함
        - MC --> sample O     --> samle backups
        - DP --> sample X     --> full backups
        - TD --> sample O     --> samle backups

- Exhaustive search --> deep & full backups --> 다 해보는 거라서 학습이라고 하기 좀 그래.. (바보같은 방법)

## TD($\lambda$) - Backward View

### Eligibilisy Tracea

- 일종의 책임 사유를 묻는 것
- 어느 state를 방문할 때마다 2가지 관점에서 책임사유 물음
    - 현재성 (가장 최근에 방문했는지)
    - 빈도 (얼마나 여러번 방문했는지)
- 방문할 때마다 1을 더해주고 시간이 지나면서 $\gamma$를 곱해줘서 감쇠 시킴
