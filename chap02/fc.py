import numpy as np


class FullyConnectedLayer:
    """
    全結合（Dense / Linear）層の最小実装（NumPy版）。

    役割：
      入力 X に対して線形変換を行う。

        Y = XW + b

    記号とshape：
      - X: 入力（ミニバッチ）           shape = (N, input_dim)
      - W: 重み行列                     shape = (input_dim, output_dim)
      - b: バイアス                      shape = (output_dim,)
            ※NumPyのブロードキャストにより (N, output_dim) に拡張されて足し込まれる
      - Y: 出力（ミニバッチ）           shape = (N, output_dim)
      - N: バッチサイズ

    画像認識での文脈：
      CNNなどで抽出した特徴ベクトル（埋め込み）をクラススコアに写像する「分類ヘッド」としてよく使う。
      例：Conv → ... → GlobalAvgPool/Flatten → FullyConnected → Softmax
    """

    def __init__(self, input_dim, output_dim):
        """
        パラメータ（W, b）の初期化。

        weights（W）:
          - np.random.randn で標準正規分布 N(0,1) からサンプルして初期化している。
          - ただし input_dim が大きいと出力の分散が過大になりやすく、
            勾配爆発/勾配消失の原因になることがある。

          - 実務・研究では分散を調整した初期化が一般的：
              Xavier/Glorot: Var(W) ≈ 1 / input_dim （tanh, sigmoidなどで安定しやすい）
              He:           Var(W) ≈ 2 / input_dim （ReLU系で安定しやすい）
            本コードは「仕組みを最小で理解する」ため単純な初期化にしている。

        bias（b）:
          - ゼロ初期化が一般的。重みがランダムなので対称性が壊れ、学習が進む。

        ここで作る変数：
          - self.grad_weights, self.grad_bias: backwardで計算した勾配を保持（updateで使用）
          - self.X: forward入力をキャッシュ（backwardで dL/dW を計算するために必要）
        """
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros(output_dim)

        self.grad_weights = None
        self.grad_bias = None
        self.X = None

    def forward(self, X):
        """
        順伝播（forward）。

        入力:
          X shape = (N, input_dim)

        出力:
          Y shape = (N, output_dim)

        計算：
          Y = XW + b

        なぜ X を保存するか：
          逆伝播で重みの勾配 dL/dW を求める式に X が登場するため。
          （計算グラフの「中間値キャッシュ」に相当）
        """
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, grad_output):
        """
        逆伝播（backward）。
        「損失 L の出力Yに対する勾配 dL/dY = grad_output」が与えられたときに、
        入力Xとパラメータ(W,b)に対する勾配を計算する。

        入力:
          grad_output = dL/dY   shape = (N, output_dim)

        出力:
          grad_input  = dL/dX   shape = (N, input_dim)

        理論（行列微分）：
          Y = XW + b とする。
          1) 入力への勾配
             dL/dX = dL/dY * dY/dX
                   = grad_output * W^T
             shape: (N, output_dim) @ (output_dim, input_dim) = (N, input_dim)

          2) 重みへの勾配
             dL/dW = X^T * grad_output
             shape: (input_dim, N) @ (N, output_dim) = (input_dim, output_dim)

          3) バイアスへの勾配
             b は各サンプルに同じだけ足されるので、
             dL/db はバッチ方向に総和を取る：
             dL/db = sum_n dL/dY_n
             shape: (output_dim,)

        直感：
          - Wの各要素は「入力の各次元が出力の各次元にどれくらい効くか」を表すので、
            X（入力）とgrad_output（出力側から来た誤差）の“相関”で勾配が決まる。
          - bは各出力ユニットに一律に足されるため、誤差をバッチ方向に足し合わせるだけでよい。
        """
        # dL/dX = (dL/dY) W^T
        grad_input = np.dot(grad_output, self.weights.T)

        # dL/dW = X^T (dL/dY)
        self.grad_weights = np.dot(self.X.T, grad_output)

        # dL/db = sum over batch
        self.grad_bias = np.sum(grad_output, axis=0)

        return grad_input

    def update(self, learning_rate):
        """
        パラメータ更新（SGDの1ステップ）。

        更新式：
          W <- W - η * dL/dW
          b <- b - η * dL/db
        ここで η = learning_rate（学習率）。

        注意点（理論的な観点）：
          - 学習率が大きすぎると発散しやすい（最急降下のステップが過大）。
          - 小さすぎると収束が遅い。
          - 実務では Adam / SGD+Momentum / Weight Decay などを使うことが多いが、
            本コードは「最小の原理理解」を目的にSGDのみ実装している。

        実装上の注意：
          - backward() を呼んで self.grad_weights / self.grad_bias が計算されてから update() を呼ぶ想定。
          - もし None のまま update を呼ぶとエラーになるので、運用上はガードを入れることもある。
        """
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias
