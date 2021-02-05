class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (H - self.pool_w) / self.stride)

        # 전개(1): 입력 데이터를 전개한다.
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 최댓값(2): 행별 최댓값을 구한다.
        out = np.max(col, axis=1)

        # 성형(3): 적절한 모양으로 성형한다.
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out