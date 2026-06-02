# 스마트모빌리티공학실험2 Final Project Report 12223636안승한

## 1. 모티베이션 & 인트로

본 프로젝트의 목표는 18개 Wi-Fi RTT 앵커의 거리 측정값을 이용하여 실내 사용자 위치를 추정하는 것이다. 단순 LS 기반 위치 추정은 모든 앵커의 측정값을 동일하게 신뢰하지만, Wi-Fi RTT 데이터에는 앵커별 평균 bias, 실내 반사, NLOS, 순간적인 측정 튐 현상이 함께 포함되어 있다. 따라서 단순 LS만으로는 안정적인 위치 추정이 어렵다.

또, 각 앵커별로 상태가 각기 달라 오차특성이 많이 달랐다는 실험경험에 기인하여 각 앵커별로 오차를 우선 보정하는 Anchor-wise calibration을 먼저 진행하였다.
이 결과를 baseline으로 하였고, 결과를 보니 여전히 잔차가 큰 outlier에 대해 오차를 줄이지 못하였다.

그러므로 두번째 단계로 잔차에따라 보정 가중치를 상이하게 적용하여 반복하는 IRWLS를 적용하였다.
Tukey 방식과 Huber 방식을 비교하였고 그 결과 Huber 방식의 성능이 우수하여 이를 선택하였다.

세번째로는 머신러닝을 적용하여 추가적인 성능개선을 이끌어내고자 하였다.
RandomForest, ExtraTrees 방식에 대해 처음부터 머신러닝만 쓰는 방식과 Calibrate + IRWLS를 적용한 2차버전의 결과에 남은 잔차를 보정하는 용도로 쓰는 방식 총 4가지를 수행하였고, 이 결과 처음부터 머신러닝만 쓰는 방식은 결과가 좋지 않았다.

마지막으로, 더 개선할 방법을 찾아보던 중 HistGradientBoosting 방법에 Confidence Gate를 결합하면, Huber IRWLS 결과의 신뢰도에 따라 보정 강도를 조절할 수 있다는 사실을 알아냈다. Boosting 계통 방법은 이전 tree가 설명하지 못한 오차를 다음 tree가 보정하는 방식이므로 IRWLS 이후 남아있는 위치 잔차를 단계적으로 줄이는 데 효과가 있을 것 같아 이를 최종적으로 선택했다.

즉, 결론적으로 본 방법은 Anchor-wise Calibration + Huber 기반 IRWLS + HisGradientBoosting with Rich Feature + Gate Scale을 적용한 것이다. 
 오차를 한 번에 해결하려 하지 않고, 오차 원인을 단계별로 분리하여 처리하였으며 최종 알고리즘은 `AC-HGBoost-Gate`로 정의하였다. 이는 `Anchor-wise Calibrated Huber-initialized Rich Feature Gated HistGradientBoosting`의 약자이다.



차별점은 머신러닝을 처음부터 위치를 예측하는 black-box 모델로 사용하지 않는 것이다. 먼저 물리 기반 위치 추정으로 초기 위치를 계산하고, HGB는 그 결과에서 남은 잔차만 보정한다. 이를 통해 물리적 위치 추정의 해석 가능성과 데이터 기반 보정의 장점을 살려 조합하고자 하였다.

## 2. 알고리즘 설명

### 2.1 Anchor-wise Calibration

RTT 측정값은 앵커별로 일정한 방향의 bias를 가질 수 있다. 따라서 전체 거리값을 하나의 상수로 보정하지 않고, 각 앵커마다 독립적인 평균 bias를 계산하였다.

사용자 위치를 p, i번째 앵커 위치를 b_i, 측정 거리를 d_hat_i라고 하면 실제 거리는 ||p - b_i||이다. 앵커별 bias는 d_hat_i - ||p - b_i||의 평균으로 계산하였다. Validation 실험에서는 train set에서만 bias를 계산하고 validation set에 적용하여 정답 정보가 새어 들어가지 않도록 하였다.

Anchor-wise Calibration 적용 결과, Raw LS 대비 MAE가 22.186 m에서 9.554 m로 감소하였다. 이는 제공 데이터에서 앵커별 systematic bias가 큰 영향을 갖고 있음을 보여준다.

### 2.2 Huber IRWLS

Calibration 이후에도 일부 RTT 측정값의 잔차는 거의 줄어들지 않았다. 이처럼 과도한 outlier는 일반 LS 또는 WLS에서 위치 추정 결과를 크게 왜곡할 수 있으므로 이를 완화하기 위해 Huber 기반 IRWLS를 적용하였다.

현재 위치 추정값을 p_huber, 보정된 거리값을 d_cal_i라고 하면 i번째 앵커의 residual은 ||p_huber - b_i|| - d_cal_i로 정의하였다. Residual의 scale은 표준편차 대신 MAD 기반 robust scale을 사용하였다. Huber weight는 정규화 residual이 작을 때는 1에 가깝게 유지되고, residual이 커지면 residual 크기에 반비례하여 감소한다.

본 실험에서는 Huber threshold를 안정적인 기본값인 1.345로 설정하였다. Huber IRWLS는 큰 residual을 가진 앵커를 완전히 제거하지 않고 영향만 줄이므로, 앵커 수가 제한된 상황에서도 정보 손실을 과도하게 만들지 않는 장점이 있다.

### 2.3 Rich Feature Gated HGB Residual Correction

3차 개선은 HistGradientBoosting 기반 residual correction이다. HGB는 이전 tree가 설명하지 못한 오차를 다음 tree가 순차적으로 보정하는 boosting 계열 모델이다. 따라서 Huber IRWLS 이후에도 남아 있는 위치 residual을 단계적으로 줄이는 문제와 구조적으로 잘 맞는다.

HGB의 target은 최종 위치 자체가 아니라 Huber IRWLS의 위치 오차이다. 즉, target은 p_true - p_huber로 정의하였다. HGB는 Huber 초기 위치에서 얼마나 이동해야 실제 위치에 가까워지는지를 학습한다.

최종 위치는 p_final = p_huber + 1.1 × delta_HGB로 계산하였다. 여기서 1.1은 validation 실험에서 선택된 residual gate scale이다. 이 gate는 HGB가 예측한 residual 보정량을 그대로 더하지 않고, 보정 강도를 조절하기 위해 사용하였다.

### 2.4 Rich Feature 구성

HGB 입력으로는 calibrated RTT 거리 외에도, 2차 개선단계의 Huber IRWLS에서 생성된 residual, weight, predicted distance, residual 통계량을 함께 사용하였다. 이를 통해 HGB가 단순 거리-좌표 관계뿐만 아니라 현재 추정 결과가 어떤 앵커에서 불안정한지도 학습하도록 하였다.

| Feature | 개수 | 의미 |
| Raw RTT distance | 18 | 원본 RTT 거리값 |
| Calibrated RTT distance | 18 | 앵커별 bias가 제거된 거리값 |
| Huber initial position | 2 | Huber IRWLS로 얻은 초기 x, y 위치 |
| Anchor residual | 18 | 예측 거리와 보정 거리의 차이 |
| Absolute residual | 18 | residual의 절댓값 |
| Residual ratio | 18 | residual / calibrated distance |
| Huber final weight | 18 | 각 앵커에 부여된 최종 Huber 가중치 |
| Predicted anchor distance | 18 | Huber 위치 기준 앵커까지의 예측 거리 |
| Residual and weight statistics | 8 | residual/weight의 평균, 중앙값, 최대값, 표준편차 등 |

위와 같이 총 136개의 입력 feature를 설계하였다.

HGB는 거리값만 보는 것이 아니라 Huber IRWLS가 만든 추정 과정의 내부 정보를 함께 사용하므로 어떤 앵커의 residual이 큰지, 어떤 샘플에서 weight가 불안정한지, Huber 초기 위치가 얼마나 신뢰 가능한지를 반영하여 residual을 보정할 수 있다.

## 3. Agent AI 활용 방안

Chat GPT를 사용하였으며, 본 프로젝트의 실험단계인 앵커별 캘리브레이션, IRWLS, 머신러닝을 구성하는 것은 스스로 하였다.
 다만 각 단계 수행 결과를 분석하는 것은 AI를 사용하였으며, 해당 결과에 적합한 다음단계 구현방법을 제시하라고 명령하였다. 예를 들어 "IRWLS를 적용해야겠는데 구현하는 방법으로는 어느것이 있으며, 각 방법의 특징을 알려달라"는 방식이다.
  본인은 이러한 제시결과를 토대로 가장 적합한 방법론을 선택하였고, 해당 알고리즘을 구현, validation 결과를 해석하는 데 AI를 사용한 것이다.

  이외에 수식정리, 코드 구조화, feature 설계, 결과 해석 초안 작성에 AI를 사용하였다.
  즉, AI를 통해 최대한 많은 방법을 구하고, 그 중 어떤 방법을 쓸지에 대해서는 본인이 결정하였으며 해당 방법을 구현하는 데 AI를 사용하였다. 

구체적으로 AI는 세 단계에서 활용되었다. 1. LS baseline 이후 남는 오차가 단순 random noise가 아니라 앵커별 bias와 outlier를 포함한다는 점을 정리하는 데 사용하였다. 2. Huber IRWLS 이후의 residual을 머신러닝으로 직접 보정하는 구조를 설계하는 데 활용하였다. 3. 단순 distance feature 대신 residual, weight, residual ratio, predicted distance, 통계량을 포함한 rich feature 구성을 만드는 데 활용하였다. 4. 보고서 초안 중 결과, 구성같이 데이터에 기반하여 작성해야 하는 부분에 사용하였다

최종 선택은 AI의 추천이 아니라 validation 결과를 기준으로 본인이 판단하였다.

## 4. 결과 도출 & 디스커션

### 4.1 Ablation Study

동일한 validation split을 사용하여 단계별 성능을 비교하였다.

| Method | MAE (m) | RMSE (m) | Median Error (m) | 90% Error (m) | Max Error (m) |
| Raw LS | 22.186 | 23.724 | 20.023 | 31.855 | 64.314 |
| Anchor-wise Calibrated LS | 9.554 | 11.986 | 7.764 | 17.804 | 63.214 |
| Calibrated + Huber IRWLS | 8.431 | 9.738 | 7.560 | 14.747 | 26.141 |
| Final Rich Feature Gated HGB | 5.810 | 7.244 | 4.886 | 11.165 | 23.874 |

Raw LS는 앵커별 거리 bias와 outlier를 별도로 고려하지 않기 때문에 MAE가 22.186 m로 크게 나타났다. Anchor-wise Calibration 적용결과 MAE가 9.554 m로 감소하였으며, 이는 데이터 내에 앵커별 systematic bias가 존재했음을 의미한다고 볼 수 있다. 이후 Huber IRWLS를 적용하면 MAE는 8.431 m로 추가 감소하고, 최대오차는 63.214 m에서 26.141 m로 크게 감소하였다. 이는 Huber weighting이 큰 residual을 가진 앵커의 영향을 줄여 outlier 안정성을 개선했기 때문으로 해석된다.

마지막으로 Rich Feature Gated HGB를 적용하면 MAE는 5.810 m, Median Error는 4.886 m, 90% Error는 11.165 m까지 감소하였다. 이는 Huber IRWLS 이후에도 남아 있는 residual pattern을 HGB가 추가적으로 학습하여 보정했기 때문이라고 사료된다.

### 4.2 최종 개선율

최종 AC-HGBoost-Gate 모델은 baseline 대비 모든 주요 지표에서 큰 오차 감소를 보였다.

| Metric | Baseline | Final AC-HGBoost-Gate | Reduction |
| MAE | 22.186 m | 5.810 m | 73.8% |
| RMSE | 23.724 m | 7.244 m | 69.5% |
| Median Error | 20.023 m | 4.886 m | 75.6% |
| 90% Error | 31.855 m | 11.165 m | 65.0% |
| Max Error | 64.314 m | 23.874 m | 62.9% |

특히 Median Error는 일반적인 사용자에 대한 위치추정 성능을 보여주므로 해당 사항이 4.886 m로 감소한 점이 의미있다. 이는 일반적인 사용자 구간에서 최종 모델이 안정적인 보정 성능을 보였음을 의미한다. 또한 90% Error도 11.165 m로 감소했으므로 대부분의 사용자에 대해 오차가 baseline보다 크게 줄어들었음을 알 수 있다.

### 4.3 Baseline 비교의 fairness

baseline은 Raw LS로 설정하였다. 이는 RTT 기반 다변측량에서 가장 기본적인 위치 추정 방식이며, 별도의 bias correction, robust weighting, machine learning correction을 적용하지 않은 기준 모델이다. 따라서 제안 방법의 각 개선 단계가 실제로 어느 정도 성능 향상에 기여했는지를 확인하기 위한 기준으로 적합하다.

또한 본 방법은 Raw LS와 완전히 다른 입력 정보를 사용하지 않는다. 모든 방법은 동일한 d_hat 거리 측정값과 동일한 앵커 위치 정보를 사용하므로 차이는 오직 동일한 입력을 어떻게 처리하느냐에 있다. 따라서 단순히 더 많은 센서나 외부 정보를 사용한 것이 아니라, 동일한 RTT 데이터에 대해 calibration, robust weighting, residual correction을 적용한 것이므로 baseline과의 비교는 fair하다고 볼 수 있다.

평가 과정에서도 validation set의 정답 위치가 학습 과정에 직접 사용되지 않도록 하였다. Anchor-wise bias는 train set에서만 계산한 뒤 validation set에 적용하였고, HGB 역시 train set의 residual을 학습한 뒤 validation set에서 성능을 평가하였다.

### 4.4 디스커션

AC-HGBoost-Gate의 장점은 물리 기반 추정과 boosting 기반 residual 보정이 역할을 분담한다는 점이다. Calibration은 앵커별 평균 bias를 제거하고, Huber IRWLS는 outlier에 robust한 초기 위치를 제공한다. 이후 HGB는 rich feature를 이용하여 남은 residual을 순차적으로 보정한다.

이 방식은 단순히 거리값을 이용해 위치를 직접 예측하는 방법보다 해석 가능성이 높다. 또한 residual, weight, residual ratio, predicted distance를 함께 사용하므로, 각 샘플에서 어떤 앵커가 위치 추정을 어렵게 만드는지에 대한 정보도 feature에 반영된다.

한계의 경우, Anchor-wise bias는 제공된 labeled data에서 계산한 평균값이므로 hidden test의 환경 분포가 달라질 경우 bias 보정 효과가 줄어들 수 있다. 또한 HGB는 train 데이터의 residual pattern을 학습하므로, 완전히 다른 공간 분포나 앵커 오차 특성에서는 성능이 달라질 수 있다는 것이다.
 무엇보다도, 사실 적합한 머신러닝 방법을 결정하는 과정에서 HGB 대신 널리 알려져있는 RandomForest를 사용한 적이 있었는데 이 방법의 MAE 결과가 본 방법보다 좋았다는 것이다. 본 방법의 경우 AI로는 돌릴 수 없을 정도로 무거웠기 때문에 사실 성능을 고려해볼때 합리적인 방법이라고 하긴 어렵다는 것이다.
 
대신 본 방법은 RandomForest에 비해 Median Error와 90% Error에 대해 더 우수한 결과를 냈으므로 구현난이도나 코드의 무게를 제외한다면 무조건 나쁘다고 할 수는 없는 방법이다. 무엇보다도 독창적이라고 생각한다.

## 5. Reference

본 프로젝트는 수업에서 진행한 LS, WLS, robust weighting 기반 위치 추정 개념과 제공 데이터에 대한 자체 validation 실험을 기반으로 작성하였으며, 최종 알고리즘과 모델 선택은 제공 데이터에서 수행한 validation 결과를 기준으로 결정하였다.
 추가적으로 참고한 자료는 다음과 같다.

[1] P. J. Huber, “Robust Estimation of a Location Parameter,” The Annals of Mathematical Statistics, vol. 35, no. 1, pp. 73–101, 1964.
본 프로젝트의 Huber IRWLS 단계에서 residual이 큰 앵커의 영향을 완화하는 robust weighting 개념을 설명하기 위한 이론적 배경으로 참고하였다. Huber의 robust estimation은 이상치에 의해 추정값이 과도하게 흔들리는 문제를 줄이는 대표적인 방법이다.

[2] J. H. Friedman, “Greedy Function Approximation: A Gradient Boosting Machine,” The Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.
본 프로젝트의 Rich Feature HGB residual correction 단계에서 사용한 boosting 기반 순차적 residual 보정 개념의 배경으로 참고하였다. Gradient boosting은 이전 모델이 설명하지 못한 오차를 다음 모델이 단계적으로 보정하는 방식이므로, Huber IRWLS 이후 남은 위치 residual을 줄이는 구조와 연결된다.

[3] Scikit-learn developers, “HistGradientBoostingRegressor,” scikit-learn documentation.
최종 모델의 HGB 구현은 scikit-learn의 HistGradientBoostingRegressor를 기반으로 하였다. 본 프로젝트에서는 해당 모델을 직접 위치를 예측하는 black-box로 사용하지 않고, p_true - p_huber residual을 예측하는 보정기로 사용하였다.

[4] K. Kosek-Szott, S. Szott, W. Ciezobka, M. Wojnara, K. Rusek, and J. Segev, “Indoor Positioning with Wi-Fi Location: A Survey of IEEE 802.11mc/az/bk Fine Timing Measurement Research,” 2025.
Wi-Fi RTT/FTM 기반 실내측위의 전반적인 배경과 RTT 거리 측정 기반 위치 추정 문제를 이해하기 위한 참고 자료로 사용하였다. 본 프로젝트는 제공된 Wi-Fi RTT 거리 측정값을 이용해 실내 위치를 추정하는 문제이므로, 해당 survey는 기술적 배경을 설명하는 데 관련된다.
