## SVM이란 무엇인가?

SVM이란, Support Vector Machine이다.

여기서 Support Vector가 무엇인지 살펴보자.

머신러닝 혹은 선형대수, 아니면 하다못해 수학을 공부하다 보면 벡터라는 놈이 계속 등장한다. 이 용어에 대해서 알고 넘어가는 것이 앞으로의 공부에 도움이 될 것이다.

스칼라와 벡터, 매트릭스는 선형대수에서 끊임없이 등장하는 하나의 약속이니 잘 알아두도록 하자.

* Scalar : 하나의 숫자 (e.g., 1,4,5...)
* Vector : 하나의 배열(e.g., [1,3],[1,5]..)
* Matrix : 2차원 행렬을 의미한다.꼭 2x2 크기의 행렬이 아니더라도, N x M(단, N,M 둘중 어느 하나가 1이 되어서는 안된다. 1이 되는 그것은 벡터로 간주한다.)형태는 행렬, Matrix로 간주한다.

---

자 이제 여기까지 기초적인 스칼라, 벡터 그리고 행렬에 대해서 알아보았다.

그렇다면 SVM은 도대체 무엇일까?

결론부터 얘기하자면 SVM은 분류를 위한 선을 정의하는 모델이다.

