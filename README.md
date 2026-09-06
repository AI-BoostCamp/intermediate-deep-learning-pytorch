# PyTorch 기반 Deep Learning 중급

<p class="gh-only">📖 <b>웹 교재로 보기:</b> <a href="https://ai-boostcamp.github.io/intermediate-deep-learning-pytorch/">https://ai-boostcamp.github.io/intermediate-deep-learning-pytorch/</a></p>

**Deep Learning 의 개념을 용어 중심으로 정리한 뒤, PyTorch(Lightning) 로 모델을 설계·학습하는 workflow 를
익히고, 대표적인 ConvNet(AlexNet · VGG · GoogLeNet · ResNet · DenseNet), 순환 신경망(LSTM · GRU, 시계열 ·
Char-RNN · 손글씨 생성), 생성 모델(Auto Encoder · VAE · GAN 계열)까지** 다루는 교재다. 강의 "PyTorch 기반
Deep Learning 중급"(홍근선, ㈜한국AI연구소)의 슬라이드(Rev. 25.10)를 읽는 글로 다시 쓴 것으로, 모든 실습은
Google Colab 노트북(PyTorch · PyTorch Lightning)과 함께 제공된다.

1장은 딥러닝의 핵심 용어와 최근 동향을 정리하고, 2장은 PyTorch 와 Lightning 의 모델 설계 workflow · 데이터셋
API · 핵심 API 를 익힌다. 3장과 4장은 ConvNet 의 원리와 대표 구조, 최적화·정규화 기법, feature map 시각화 ·
transfer learning · CAM 실습을 다룬다. 5장은 RNN cell 과 모델 구조에서 시작해 시계열 예측 · 문자 생성 ·
MNIST-RNN · Mixture Density Network 손글씨 생성으로 나아가고, 6장은 Auto Encoder 에서 변분 추론과 VAE, GAN ·
DCGAN · CGAN · ACGAN · infoGAN 까지 생성 모델을 구현한다. 장 번호는 강의 슬라이드를 그대로 따른다.

## 목차

**[교재 개요와 전체 목차](00_교재개요.md)** — 강의 전체 목차, 실습 번호 규칙

**1장 Deep Learning 개념 정리** — [장 개요](ch01_DL개념정리/00_1장개요_장개요.md)

- [1.1 Deep Learning 개요](ch01_DL개념정리/1.1_DL개요.md) — 지식·지능과 인공지능, 기계학습에서 딥러닝으로,
  뉴런과 회귀 · 로지스틱 회귀 · Softmax, 용어 정리, inductive bias 와 표현 학습, 최근 동향(생성 모델 · LLM ·
  추론 모델 · AI 하드웨어와 연산 정밀도)
- [1.2 Neural Network Simulation](ch01_DL개념정리/1.2_NN시뮬레이션.md) — TensorFlow Playground 로 활성화
  함수 · 층 수 · 특징이 모델 표현력에 미치는 영향 관찰
- [1.3 Optimization, Regularization, Learning rate](ch01_DL개념정리/1.3_최적화와정규화.md) — 최적화
  알고리즘, 학습률과 batch size, overfitting 과 L1 · L2 regularization
- [1장을 마치며](ch01_DL개념정리/99_1장마무리.md) — 세 절을 잇는 정리와 자기 점검

**2장 PyTorch 핵심 정리** — [장 개요](ch02_PyTorch핵심정리/00_2장개요_장개요.md)

- [2.1 Model 설계 workflow](ch02_PyTorch핵심정리/2.1_Model설계workflow.md) — PyTorch Lightning 의 다섯 단계,
  LightningModule · DataModule · Trainer, loss · metric, optimizer 와 lr scheduling, TensorBoard · callback,
  모델 저장과 복원 (실습 2.1.1 · 2.1.2)
- [2.2 Torchvision datasets API](ch02_PyTorch핵심정리/2.2_TorchvisionDatasets.md) — 내장 데이터셋과
  DataLoader, 같은 클래스 이미지의 다양성 (실습 2.2.1)
- [2.3 PyTorch APIs](ch02_PyTorch핵심정리/2.3_PyTorchAPIs.md) — Tensor · computation graph · Parameter,
  Layers API, activation 함수 비교, Dropout · BatchNorm · weight decay (실습 2.3.1 · 2.3.2)
- [2.4 TensorFlow(Keras) vs PyTorch](ch02_PyTorch핵심정리/2.4_Keras대PyTorch.md) — 두 프레임워크의 차이와
  Keras 코드의 기본 구조

**3장 Convolutional Neural Network(1)** — [장 개요](ch03_CNN1/00_3장개요_장개요.md)

- [3.1 ConvNet 개요](ch03_CNN1/3.1_ConvNet개요.md) — 영상인식의 어려움, LeNet 과 기본 구조, convolution 연산과
  receptive field, MLP · CNN 비교 (실습 3.1.1)
- [3.2 AlexNet / VGGNet](ch03_CNN1/3.2_AlexNet_VGGNet.md) — ImageNet 과 ILSVRC, AlexNet 의 기여(ReLU · LRN ·
  augmentation · dropout), VGGNet 의 3×3 filter 와 깊이 연구 (실습 3.2.1)
- [3.3 Optimization](ch03_CNN1/3.3_Optimization.md) — 가중치 초기화, hyper parameter 탐색, SGD 에서 Adam ·
  AdamW · Lion 까지, generalization 과 overfitting 대책(augmentation · weight decay · early stopping · ensemble ·
  dropout), Batch/Layer/Instance normalization, imbalanced data 와 semi-supervised learning (실습 3.3.1)
- [3.4 실습 : CNN model handling](ch03_CNN1/3.4_CNN모델핸들링.md) — Modern CNN 튜닝, feature map 시각화,
  pre-trained VGG 의 feature map (실습 3.4.1 · 3.4.2 · 3.4.3)
- [3.5 실습 : Image Dataset handling](ch03_CNN1/3.5_이미지데이터셋핸들링.md) — 폴더로 구분된 이미지
  데이터셋, ImageFolder · DataLoader, transforms.v2 와 real-time augmentation (실습 3.5.1)
- [3장을 마치며](ch03_CNN1/99_3장마무리.md)

**4장 Convolutional Neural Network(2)** — [장 개요](ch04_CNN2/00_4장개요_장개요.md)

- [4.1 GoogLeNet](ch04_CNN2/4.1_GoogLeNet.md) — Inception module 과 1×1 convolution, 보조 분류기, Inception
  v2 · v3 · v4, GoogLeNet 코딩과 flower photos 학습 (실습 4.1.1)
- [4.2 ResNet](ch04_CNN2/4.2_ResNet.md) — 깊은 망의 문제와 residual learning, pre-activation · bottleneck
  block, ensemble 해석, ResNet 모델 만들기 (실습 4.2.1)
- [4.3 DenseNet](ch04_CNN2/4.3_DenseNet.md) — Dense block 과 concatenation, DenseNet-BC, ResNet 과의 비교
- [4.4 Weakly Supervised Learning](ch04_CNN2/4.4_WeaklySupervised.md) — 약한 지도학습, classifier 로
  localization 하기, Class Activation Map 과 응용
- [4.5 Transfer Learning 실습](ch04_CNN2/4.5_TransferLearning.md) — 사전학습 VGG16 을 CIFAR10 · custom
  dataset 에 fine tuning, InceptionV3 fine tuning (실습 4.5.1 · 4.5.2 · 4.5.3)
- [4.6 Class Activation Map(CAM) 실습](ch04_CNN2/4.6_CAM.md) — Grad-CAM 의 원리, forward/backward hook,
  CAM 이미지 만들기 (실습 4.6.1)
- [4장을 마치며](ch04_CNN2/99_4장마무리.md)

**5장 Recurrent Neural Network** — [장 개요](ch05_RNN/00_5장개요_장개요.md)

- [5.1 RNN Cell Architectures](ch05_RNN/5.1_RNNCell.md) — 순환신경망의 개념과 사례, Basic RNN · LSTM · GRU
  cell, BPTT 와 TBPTT
- [5.2 RNN Model Architectures](ch05_RNN/5.2_RNNModel.md) — one-to-one 에서 many-to-many 까지, TimeDistributed,
  LSTM AutoEncoder, parallel · stacked · bidirectional LSTM, seq2seq (실습 5.2.1)
- [5.3 RNN Model Design (Time Series)](ch05_RNN/5.3_TimeSeries.md) — 주가 데이터의 window 구성, basic ·
  stacked LSTM 회귀, indicator 추가와 attention (실습 5.3.1 · 5.3.2 · 5.3.2.2)
- [5.4 RNN Model Design (Char-RNN)](ch05_RNN/5.4_CharRNN.md) — 문자 indexing, many-to-one 과 many-to-many
  문자 생성 모델 (실습 5.4.1)
- [5.5 RNN Model Design (MNIST-RNN)](ch05_RNN/5.5_MNIST_RNN.md) — 이미지를 시퀀스로 읽는 LSTM,
  bidirectional LSTM, Hierarchical RNN (실습 5.5.1 · 5.5.2)
- [5.6 Handwriting Generation with RNN](ch05_RNN/5.6_HandwritingGen.md) — inverse problem 과 Mixture Density
  Network, MLE 와 cost function, handwriting prediction · synthesis (실습 5.6.1)
- [5장을 마치며](ch05_RNN/99_5장마무리.md)

**6장 Generative Model** — [장 개요](ch06_GenerativeModel/00_6장개요_장개요.md)

- [6.1 Auto Encoder / Denoise Auto Encoder](ch06_GenerativeModel/6.1_AutoEncoder.md) — encoder · decoder 와
  잠재 공간, denoising, MLP · CNN · 2차원 잠재공간 AE (실습 6.1.1 · 6.1.2 · 6.1.3)
- [6.2 Variational Inference](ch06_GenerativeModel/6.2_VariationalInference.md) — 잠재 변수, entropy · cross
  entropy · KL divergence, ELBO
- [6.3 Variational Auto Encoder(VAE)](ch06_GenerativeModel/6.3_VAE.md) — reparameterization trick,
  reconstruction error 와 regularization, 잠재 영역 시각화 (실습 6.3.1)
- [6.4 GAN: vanilla GAN, DCGAN](ch06_GenerativeModel/6.4_GAN.md) — 적대적 생성 모델의 구조와 비용 함수,
  z-space, vanilla GAN 과 DCGAN 구현 (실습 6.4.1 · 6.4.2)
- [6.5 Improved GAN: CGAN, ACGAN, infoGAN](ch06_GenerativeModel/6.5_ImprovedGAN.md) — 조건부 생성과
  정보 이론적 규제 (실습 6.5.1 · 6.5.2 · 6.5.3)
- [6장을 마치며](ch06_GenerativeModel/99_6장마무리.md)

## 실습 안내

각 실습 제목 아래의 **"Open in Colab" 배지**를 누르면 노트북이 Colab 에서 바로 열린다. 노트북 파일은 이 저장소의
[`code/`](https://github.com/AI-BoostCamp/intermediate-deep-learning-pytorch/tree/main/code) 폴더에 교재의 실습
번호와 같은 이름으로 들어 있다(`실습 3.4.2` ↔ `code/3.4.2.FeatureMap.ipynb`). 절 번호만 있던 노트북에는 `.1` 을
붙였다(`실습 4.1.1` ↔ `code/4.1.1.GoogleNet.ipynb`).

실행 전 준비:

- **환경 점검** — [`code/0.0.test.ipynb`](code/0.0.test.ipynb) 로 Colab 의 Python · PyTorch · GPU 를 먼저
  확인한다.
- **런타임** — 2장과 5장의 작은 모델은 CPU 로도 되지만, 3장 이후의 ConvNet · transfer learning · GAN 실습은
  GPU 런타임에서 돌린다.
- **테스트 이미지** — 실습 3.4.3(pre-trained VGG feature map)과 4.6.1(CAM)은 `bird.jpg`, `4obj2.jpg` 같은
  테스트 사진을 쓴다. 노트북 첫머리의 준비 셀이 `/content/datasets/` 에 파일이 없으면 업로드 창을 띄우므로,
  새(또는 여러 물체가 든) 사진을 아무거나 올리면 된다. Google Drive 연결은 필요 없다.
- **모델 저장** — GAN 실습(6.4 · 6.5)은 학습된 generator 를 Colab 세션의 `/content/models/PT/` 에 저장한다.
  세션이 끝나면 사라지므로 남기려면 Drive 에 복사한다.
- **Weights & Biases** — 참고 노트북 `3.1.2.0.ConvNet_WnB.ipynb` 만 W&B 계정과 API 키가 필요하다. 키는
  노트북에 적지 말고 Colab 보안 비밀(secrets)이나 `wandb login` 으로 입력한다.

## 저자

**홍근선** — ㈜한국AI연구소 대표이사 · 전자계산기 기술사 (gshong@ai-camp.kr)
