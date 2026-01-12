from fc import FullyConnectedLayer
from relu import ReLU


# Three-layer MLP
class MLP:
    """
    MLP（Multi-Layer Perceptron; 多層パーセプトロン）の最小実装。

    このクラスは「全結合 → ReLU → 全結合」からなる 3-layer（入力層を含めた呼び方）構成を表す。
    一般に、画像認識では
      - 画像をそのままベクトル化（Flatten）して MLP に入れる（古典的）
      - CNN で抽出した特徴ベクトルを MLP に入れて分類する（現代的）
    といった形で使われる。

    アルゴリズム的に重要な点：
      - 線形層（FullyConnected）だけを重ねても「全体は線形変換」のままで表現力が不足する。
      - そこで ReLU のような非線形を挟むことで、ネットワークは非線形関数を近似できる。
      - この構造は「関数合成」として
            f(X) = W2 * ReLU(W1 * X + b1) + b2
        を学習しているとみなせる。

    ここで各層のパラメータ：
      - 1層目FC: W1, b1
      - 2層目ReLU: パラメータなし（ただし backward のために入力の符号情報を保持する）
      - 3層目FC: W2, b2
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        ネットワーク構造を定義する。

        input_dim:
          入力特徴次元 d_in
          例：画像を Flatten して (H*W*C) にした場合、その値が input_dim になる。

        hidden_dim:
          隠れ層次元 d_hidden
          ここが大きいほど表現力は増える傾向があるが、計算量・過学習リスクも増える。

        output_dim:
          出力次元 d_out（通常はクラス数 C）
          この MLP の forward 出力は logits（未正規化スコア）として扱うことが多く、
          その後に Softmax + CrossEntropy を適用して学習するのが定番。

        実装上の設計：
          layers を「順番のリスト」として持つことで、
          forward は順に適用、backward は逆順に適用、という
          計算グラフの流れと一致した実装になる（典型的なミニフレームワーク構造）。
        """
        self.layers = [
            FullyConnectedLayer(input_dim, hidden_dim),  # 線形変換（特徴抽出/変換）
            ReLU(),  # 非線形性の注入（表現力の源泉）
            FullyConnectedLayer(
                hidden_dim, output_dim
            ),  # 線形変換（クラススコアへ写像）
        ]

    def forward(self, X):
        """
        順伝播（forward）。

        入力：
          X shape = (N, input_dim)

        出力：
          logits shape = (N, output_dim)

        アルゴリズム（層の合成）：
          for layer in layers:
              X = layer.forward(X)

        これは関数合成として
          X1 = FC1(X0)
          X2 = ReLU(X1)
          X3 = FC2(X2)
        を順に計算しているのと同じ。

        重要：
          - backward のために、各層は forward 時の中間情報を内部に保持している
            （FCは入力X、ReLUは符号マスク相当）。
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    # compute the gradient of each layer and backprop it
    def backward(self, grad_output):
        """
        逆伝播（backward）。

        入力：
          grad_output = dL/d(logits)  shape = (N, output_dim)
          例：Softmax+CrossEntropy から返ってくる勾配がこれに相当する。

        出力：
          grad_input = dL/dX  shape = (N, input_dim)
          ネットワーク入力に対する勾配（通常は学習には不要だが、勾配検証や連結モデルで使う）。

        アルゴリズム（連鎖律の適用）：
          forward が
            X0 -> layer1 -> X1 -> layer2 -> X2 -> layer3 -> X3
          だったなら、backward は逆向きに
            dL/dX3 -> layer3.backward -> dL/dX2 -> layer2.backward -> dL/dX1 -> layer1.backward -> dL/dX0
          と流れる。

        実装は layers を reversed して順に backward しているだけだが、
        これは計算グラフにおける連鎖律をプログラムでなぞっている。

        注意：
          - FC層は backward の際に「パラメータ勾配（dL/dW, dL/db）」も内部に保存する。
          - ReLU は勾配をマスクするだけ（負領域の勾配を遮断）でパラメータ勾配はない。
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    # update the weights of each layer
    def update(self, learning_rate):
        """
        パラメータ更新（SGD）を各層に適用する。

        learning_rate:
          学習率 η

        アルゴリズム：
          - backward で各 FC 層に蓄えられた勾配（grad_weights, grad_bias）を使って
            W <- W - η * dL/dW
            b <- b - η * dL/db
            を行う。

        なぜ isinstance でFCだけ更新するか：
          - ReLU は学習パラメータを持たないため、更新対象ではない。
          - 一方で FullyConnectedLayer は重み・バイアスを持ち、更新が必要。

        実務的には：
          - Momentum/Adam/Weight Decay などに拡張する場合、
            「layer.update」ではなく「optimizer.step(params)」のように分離することも多い。
          - ただし本コードは学習の原理を掴む目的として、最小構成で整理されている。
        """
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                layer.update(learning_rate)
