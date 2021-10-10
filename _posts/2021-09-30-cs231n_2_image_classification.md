---
layout: single
title:  "CS231n Lecture 2. Image Classification"
---

# Image Classification


## Challenges in Image Classification

- **Image Classification**: 이미지를 보고 어떤 카테고리에 해당하는지 분류하는 것으로, is a core task in CV
- 사람은 간단하게 이미지를 분류하지만 컴퓨터에게 이미지는 단지 a big grid of numbers일 뿐
- 이러한 차이를 **Semantic Gap**이라 하며, is the main problem of the image classification.
- 이미지 분류에는 아래와 같은 여러 Challenges가 존재하며, 해당 문제들에 대해서 강인한 알고리즘을 만드는 것이 목표
    - Viewpoint variation
    - Illumination
    - Deformation
    - Occlustion (가려짐)
    - Background Clustter (Object가 background와 비슷한 경우)
    - Intraclass variation (한 클래스 내에서도 색, 크기, 생김새 등 다양함)


## Attempts have been made

- 기존에는 이미지의  edges를 계산하여 corners and boundaries의 특정 조합으로 클래스를 정의 후 분류함
- 하지만 아래 두가지 문제 발생
    - Super brittle
    - Another object category에 대해서 처음부터 새로 만들어야 함 (not a scalable approach) 
- 다양한 객체에 적용가능한 알고리즘을 위한 insight 중 하나는 **Data-Driven Approach**

## Data-Driven Approach

1. 이미지 데이터 수집
2. Machine Learning 알고리즘 기반 Classifier 학습
3. 학습된 모델을 새로운 이미지로 검증

- 이러한 접근 방법 기반의 복잡한 ML 알고리즘을 배우기 전에 아래 단순한 Classifier인 Nearest Neighbor 부터 살펴보자


## Nearest Neighbor

- 단순한 classifier 알고리즘으로, 크게 학습 단계와 예측 단계로 나뉨
    1. 학습: 단순히 모든 이미지 데이터와 레이블을 기억
    2. 예측: test image가 입력되면 기존 학습 데이터와 비교해서 가장 유사한 이미지의 레이블로 예측
- But, 학습할 데이터가 많아질수록 학습보다 예측하는데 더 많은 시간이 소요되므로 좋지 않은 방법

## K-Nearest Neighbors

- Nearest Neighbor 에서 가장 가까운 이웃 하나만 찾았다면, K-Nearest Neibors는 K개의 가장 가까운 이웃을 찾고 투표를 통해 그 중에서 가장 많은 득표(majority vote)를 한 레이블로 예측
- 투표 방법에는 (거리 기준 등) 여러가지 방법 있지만, 가장 쉬운 방법은 득표수만 고려하는 방법
- K=1일 때는 NN과 동일, K가 커질수록 decision boundary가 smooth하고 좋아짐

### Distance Metric



- 서로 다른 두 데이터를 비교할 때 데이터 간의 차이가 얼마인지 계산하는 **거리 척도**

- L1 (Manhatten) distance vs. L2 (Euclidian) distance
    - 가지고 있는 데이터의 좌표계에 따라서 전부 다름
- L1 distance
    - Coordinate dependncy를 가지고 있으므로 좌표계를 회전시키면 L1 distance도 변함
    - 예를들어 직원 분류 문제에서 데이터의 각 요소(급여나 근속년수와 같은)가 직원들의 다양한 특징에 영향을 주는 경우라면 이때는 L1 distance가 더 적합할지도 모름
    - 그래도 이해가 잘 안된다면 물리 개념에서 방향에 의존적인 속도와 방향에 상관없는 속력을 한번 떠올려보기

- L2 distance
    - 방향(좌표계)에 상관없이 동일한 거리를 가지는 경우 (feature의 영향이 골고루 적용됨)

- 그림에서 보다시피, 좌표축 방향으로 영향을 받는 L1에 비해 L2는 모든 방향으로 영향을 받으므로 좀 더 desicion boundary가 smooth한 경향이 있음

### Hyperparameters

- K 또는 distance metric과 같은 hyperparameter를 잘 선택하는 방법은?
    - problem&data-dependent 하므로 다양한 값을 시도 후 경험적으로 찾아야 함

### Setting Hyperparameters

- Hyperparameters 선택 시 언제, 어떤 데이터셋에 맞추어 선택을 해야하는지?
    1. 학습 시 사용한 데이터로만 좋은 성능 나오도록 선택
        - Bad idea.
    2. train/test로 나누어 학습 후 test dataset에만 좋은 성능 나오도록 선택
        - Bad idea. 또 다른 새로운 데이터에 대해서 강건성이 떨어짐
    3. train/validation/test로 나누고, 학습 후 validation set에 적합한 parameters 선택 후 test set은 최종 검증용으로만 사용
        - Better!
    4. Cross-Validataion: 5 folds로 구분된 dataset을 4개의 train용과 1개의 validate용으로 순환시키며 학습과 평가를 진행
        - 이 방식이 표준이지만, small dataset에서만 useful하며 딥러닝같은 큰 모델 학습 시에는 계산량이 많으므로 잘 사용하지 않음       

Q. test/validation set을 선정할 때 전체 데이터셋을 대표하지 못하고 편향되어 있다면 문제가 되지 않을까?  
- 실제로 문제가 될 수 있음
- 하지만 통계학적 가정으로 볼 때, 사용하는 data는 독립적이며(independently) 유일한 하나의(identically) 분포에서 나옴
- 따라서 모든 data는 동일한 분포를 따른다고 생각하고 진행


### K-Nearest Neighbor가 image classification에서 사용되지 않는 이유

- K-NN이 잘 동작하려면 image 전체 공간을 조밀하게 커버할 만큼의 충분한 트레이닝 샘플 필요   
이렇게 공간을 조밀하게 덮기 위해서 데이터 양이 늘어나게 되고 차원이 증가함에 따라 데이터의 양은 기하급수적으로 늘어남 (Curse of Dimensionality)   
따라서 모든 공간을 조밀하게 메우는 고차원의 이미지를 모으는 것도 현실적으로 불가능

- 픽셀 위에서의 distance만 같다고 다 똑같은 image가 아니며, distance가 다르다고 서로 다른 class라고 볼 수도 없음   
따라서 L2 distance와 같은 방법은 이처럼 두 이미지 사이의 perceptional distance를 구하는데 적절한 방법은 아님

## Linear Classification

### Parametric Approach: Linaer Classifier

- 모델의 가중치 파라미터 $W$를 학습하는 방법으로, Linear classifier는 **paramertric model**의 가장 단순한 형태
- 가중치 행렬 $W$를 업데이트 하는 방식이므로, 기존의 K-NN보다 더욱 빠른 학습이 가능
- Bias $b$는 특정 클래스에 대한 선호도를 나타내며, 데이터와는 독립적인 scaling offset로 데이터 불균형을 해소하는 역할로도 사용됨
- $ f(x, W) = Wx + b $ = class score

### Hard cases for a linear classifier

- Linear Classifier로 풀기 어려운 경우들 존재
    - Data를 선 하나로 분류할 수 없는 경우
    - Multimodal data와 같이 한 클래스이지만 다양한 공간에 분포한 경우
