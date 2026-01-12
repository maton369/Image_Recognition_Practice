import numpy as np


class Softmax:
    """
    Softmax 関数の最小実装（NumPy版）。

    目的：
      クラス分類の「スコア（logits）」を「確率分布」に変換する。

    入力：
      logits x（任意shape。ただし通常は (N, C) を想定）
        - N: バッチサイズ
        - C: クラス数
      logits は「未正規化スコア」で、実数のままでは確率になっていない。

    出力：
      p = softmax(x)（同shape）
        - 各サンプルについて、クラス方向に
          p_c >= 0 かつ Σ_c p_c = 1
        を満たす確率分布になる。

    数式（クラス方向に適用）：
      各サンプル n のクラス c に対して
        p_{n,c} = exp(x_{n,c}) / Σ_k exp(x_{n,k})

    重要な性質：
      - softmax は「相対差」だけが効く（平行移動不変性）：
          softmax(x) = softmax(x + const)
        これにより、数値安定化のために max を引いても結果は変わらない。
      - ただし exp は値が大きいとオーバーフローしやすいので、安定化が必須。

    画像認識の文脈：
      最終層の出力（logits）を softmax で確率にし、
      Cross Entropy（交差エントロピー）と組み合わせて学習するのが定番。
    """

    def __init__(self):
        # Softmax自体は学習パラメータを持たないため、特に初期化するものはない。
        pass

    def forward(self, input_data, axis=-1):
        """
        順伝播（forward）。

        input_data:
          logits（未正規化スコア）
          例: shape = (N, C)

        axis:
          softmax を計算する軸。
          - 通常は最後の次元（クラス次元）に対して計算するので axis=-1。

        数値安定化（超重要）：
          exp(input_data) は input_data が大きいと簡単に overflow する。
          そこで以下を使う：

            exps = exp(input_data - max(input_data))

          softmax の平行移動不変性により、結果は変わらない：
            exp(x_i - m) / Σ exp(x_k - m) = exp(x_i)/Σ exp(x_k)

        実装詳細：
          keepdims=True にすることで、ブロードキャストが自然に効き、
          (N, C) の形を保ったまま引き算・割り算ができる。
        """
        # 各サンプルごと（axis方向）に最大値を引いて exp のオーバーフローを防ぐ
        shifted = input_data - np.max(input_data, axis=axis, keepdims=True)

        # exp を取る（ここで shifted により数値安定化されている）
        exps = np.exp(shifted)

        # 正規化して確率分布にする
        output = exps / np.sum(exps, axis=axis, keepdims=True)
        return output


class CrossEntropyLoss_with_Softmax:
    """
    Softmax + Cross Entropy Loss をまとめた実装。

    なぜまとめるのが重要か（理論・数値安定性）：
      - 本来、損失は
          L = -log(softmax(x)[y])
        だが、softmax と log を別々に計算すると数値的に不安定になりやすい。
      - 実務では log-sum-exp trick を使った「log_softmax」を用いて
        より安定に計算するのが定番（PyTorch の CrossEntropyLoss はこれ）。

    ただし本実装は「理解を優先」して
      softmax を計算してから log を取る形を採用している。
      そのため log(0) を避けるために 1e-8 を足している。

    入力ラベル y の形式：
      - 現在の forward/backward は「クラスID（整数）」を想定：
          y shape = (N,), 各要素は 0..C-1
      - one-hot ベクトルを使う場合はコメントアウトの backward を利用可能
        （ただし forward 側の取り出しも one-hot 用に変える設計もありうる）
    """

    def __init__(self):
        self.softmax = Softmax()
        self.y = None  # 正解ラベル（クラスID）
        self.y_hat = None  # softmax出力（予測確率）

    def forward(self, x, y):
        """
        順伝播（forward）：損失値を計算する。

        入力：
          x: logits（未正規化スコア） shape = (N, C)
          y: 正解ラベル（クラスID）   shape = (N,)

        出力：
          loss: スカラー（ミニバッチ平均）

        交差エントロピー（クラスID版）：
          各サンプル n の損失は
            L_n = - log(p_{n, y_n})
          ここで p = softmax(x)

          バッチ平均にすると
            L = (1/N) Σ_n L_n

        実装上の注意：
          - log(0) を避けるために +1e-8 を入れている。
            ただし根本的には log_softmax を使う方が安定。
        """
        self.y = y
        self.y_hat = self.softmax.forward(x)  # p = softmax(x)

        # 正解クラスの確率 p_{n, y_n} を取り出す
        batch_indices = np.arange(self.y_hat.shape[0])
        correct_class_probs = self.y_hat[batch_indices, y]

        # -log(p) を取って平均
        loss = -np.sum(np.log(correct_class_probs + 1e-8)) / self.y_hat.shape[0]
        return loss

    # -------------------------------
    # one-hot ベクトル（y: shape=(N,C)）の場合の典型形：
    #   dL/dx = (y_hat - y) / N
    # これは「softmax + CE をまとめたとき」の有名な簡約形で、
    # 勾配計算が非常にシンプルになる（実装・計算が安定しやすい）。
    # -------------------------------
    # def backward(self):
    #     return (self.y_hat - self.y) / self.y_hat.shape[0]

    def backward(self):
        """
        逆伝播（backward）：logits x に対する勾配 dL/dx を返す。

        出力：
          dx = dL/dx   shape = (N, C)

        理論（ここが最重要）：
          softmax と cross entropy を組み合わせた場合、
          「softmax のヤコビアン（C×C）」を明示的に作らなくても、
          勾配が次の形に簡約される：

            dx = (p - t) / N

          - p = softmax(x)（予測確率）
          - t = 正解分布（one-hot。クラスIDなら、該当クラスだけ1のベクトル）
          - N = バッチサイズ

        直感：
          - 予測確率 p が正解分布 t に近づく方向へ勾配が働く。
          - 正解クラスでは p_y - 1（負方向になりやすい）→ logits を上げる方向へ更新される。
          - 非正解クラスでは p_c - 0 = p_c（正方向）→ logits を下げる方向へ更新される。

        実装の対応：
          - dx を y_hat で初期化し、正解クラスだけ 1 引くことで (p - t) を作る。
          - 最後に batch_size で割って平均損失の勾配にする。
        """
        batch_size = self.y.shape[0]

        # dx を予測確率 p で初期化
        dx = self.y_hat.copy()

        # 正解クラス成分だけ 1 を引く（p - t を作る）
        dx[np.arange(batch_size), self.y] -= 1

        # バッチ平均に対応するように 1/N を掛ける
        dx = dx / batch_size
        return dx
