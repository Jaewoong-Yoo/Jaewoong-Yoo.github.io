---
layout: single   
title: "RL Lecture 6. Value Function Approximation"   

---

# Value Function Approximation

## Incremental Methods

- 목적 함수는 $J(w) = E_\pi[(v_\pi(S)-\hat{v}(S,w))^2]$ 으로, true value $\hat{v}(S,w)$에서 approximate value $v_\pi(S)$를 뺀 값(크기)의 기댓값
- Stochastic gradient descent를 이용해서 graident에 대한 sample을 구하면,  
**$\Delta w = -1/2\alpha  = \alpha(v_\pi(S)-\hat{v}(S,w))\nabla_w\hat{v}(S,w)$**

### Linear Function Approximation

- feature vector들의 선형합(Linear Combination or Weighted sum)으로 value가 나올 것이라고 가정을 하고, 이를 이용하여 value function을 표현해 볼 것이다.
- Linear value function approximation 일 때 $\hat{v}(S,w) = x(S)^Tw$ (feature value)이므로, **$\Delta w = \alpha(v_\pi(S)-\hat{v}(S,w))x(S)$**
- 즉, Update (gradient descent에서 이동하는 정도) = step-size X prediction error (틀린 정도) X feature value
- 따라서 Update($\Delta w$)만큼 바꿔나가면서 weight 벡터 $w$를 계속 업데이트 하는 것

### Incremental Prediction Algorithms

- 이제 부터는 이전 강의에서 해왔던 방법 그대로 적용한는 것
- 만약 MC라면, target인 true value function 자리에 return $G_t$을 대체해서 넣으면되고,  
(여기서 return은 reward의 discounted cumulative sum임)
- 만약 TD(0)라면, TD target 자리에 한 step 더 간 reward에 그 step에서의 예측치의 합을 대체로 넣기
- 만약 TD($\lambda$)라면, $\lambda$-return인 $G_t^\lambda$를 대체로 넣으면 끝

$ $
- Q 질문: true value function 자리에 TD target이 들어가게 되면 TD target에 parameter $w$와 관련 변수가 있는데, 그럼 update의 $w$에 대한 미분과정이 꼬여버리는 것 아닌가??
    - 답: update할 때 시간축 기준으로 우리는 미래인 한 방향을 보고 가는데, 만약 예측값 조차 업데이트한다면 한 스텝 더 가서 과거를 보는 것과 과거에서 미래를 보는 것이 서로 섞이게 되므로 시간을 역행해버리게 되고 식이 복잡해진다. 일단 이렇게 이해해야함.


### Incremental Control Algorithms

- 어렵게 생각할 것 없이, q(action value function)는 위의 v(state value function)하고 똑같으며, given value에 $A$가 추가된다는 점 빼곤 모두 동일.


## Batch Methods

- 기존 incremental method는 sample 하나씩만보고 업데이트 후 버리기 때문에 sample사용이 효율적이지 않음. 따라서 쌓여있는 경험들(state와 value의 쌍) experience $D$에서 sampling을 하여 사용 = **Experience Replay**
- **Experience Replay**는 Off-Policy에서 굉장히 많이 쓰이는 방법
