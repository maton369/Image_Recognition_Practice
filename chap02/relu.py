import numpy as np


class ReLU:
    """
    ReLU（Rectified Linear Unit）活性化関数の最小実装（NumPy版）。

    定義：
      入力 x に対して
        f(x) = max(0, x)
      を要素ごと（element-wise）に適用する。

    画像認識での意味：
      - CNN/MLP で最も頻繁に使われる非線形。
      - 線形層（Conv/FC）だけを重ねても全体は線形写像のままだが、
        ReLUを挟むことで「非線形な決定境界」を表現できるようになる。
      - 実務的にも、sigmoid/tanhより勾配消失が起きにくく学習が進みやすいことが多い。

    数学的な性質（重要）：
      - ReLUは x<0 を 0 に潰し、x>0 をそのまま通すため、
        「負の領域では勾配が 0」になる（dead ReLU の原因）。
      - x=0 では微分が厳密には定義されない（尖点）。
        ただし実装では慣習的に「0 か 1 のどちらか」を採用する。
        NumPy等の実装では x=0 の点は測度ゼロなので、学習の実用上は大きな問題になりにくい。
    """

    def __init__(self):
        """
        ReLUは学習パラメータ（重み・バイアス）を持たない層。
        そのためコンストラクタで初期化するものは基本的にない。
        """
        self.input = None  # backwardでマスクを作るため、forwardの入力を保存する

    def forward(self, input_data):
        """
        順伝播（forward）。

        入力:
          input_data: 任意shapeの配列（例：画像特徴なら (N, C, H, W) など）
        出力:
          out = max(0, input_data) を要素ごとに適用した配列（同じshape）

        実装の要点：
          - np.maximum(0, input_data) は要素ごとに比較して大きい方を取る。
          - 逆伝播のために input_data を保存しておく。
            （ReLUの勾配は「入力が正だったかどうか」に依存するため）
        """
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, d_output):
        """
        逆伝播（backward）。

        入力:
          d_output = dL/d(out)  （上流から渡ってくる勾配）
          shape は forward 出力と同じ

        出力:
          grad_input = dL/d(input)

        理論（連鎖律）：
          out = f(input) とすると
            dL/d(input) = dL/d(out) * d(out)/d(input)

          ReLUの導関数は（要素ごとに）
            f'(x) = 1  (x > 0)
                  = 0  (x < 0)
            x=0 は未定義だが、実装では 0 とみなすことが多い。

          よって
            grad_input = d_output * 1_{input > 0}
          となる。ここで 1_{...} は条件を満たすとき1、そうでなければ0の指示関数。

        実装の要点：
          - d_output をコピーして、負領域の要素だけ 0 にする（マスク適用）。
          - self.input < 0 の場所は ReLU 出力が 0 で勾配も遮断されるため 0。
            （※ self.input == 0 の扱いはこの実装では「0にしない」ではなく、
              条件が <0 なので 0の点はそのまま残る。厳密には未定義だが、
              多くの実装では <=0 にして 0 点も遮断することが多い。
              どちらでも実用上大差は出にくいが、一貫性のため <=0 に寄せる設計もある。）
        """
        grad_input = d_output.copy()

        # 入力が負だった場所は ReLU の傾きが 0 なので、勾配を遮断する
        grad_input[self.input < 0] = 0

        return grad_input
