# 技术指标

这里是一些常见的技术指标。

这些指标不能单独使用，可以用于组成策略，或者用作模型的特征。

## SMA（简单移动平均）

简单移动平均就是最近 N 天的值之和。一般认为它是一种去噪的手段，去噪之后的值是实际值。

$SMA(t,n) = \frac{x(t) + x(t-1) + \cdots + x(t-n+1)}{n}$

```py
def sma(arr, wnd):
    res = []
    for i in range(0, len(arr) - wnd + 1):
        res.append(arr[i:i + wnd].mean())
    return np.asarray(res)
```

## EMA（指数移动平均）

EMA 也是一种移动平均，它会给最近的数值较大的权重。

要注意，EMA 的窗口并不是停止的位置，窗口之外的值也占权重。数组计算 EMA 之后长度是不变的。

$EMA(t,n) = \alpha x(t) + (1-\alpha)x(t-1)$

$\alpha = \frac{2}{n+1}$

$EMA(1, n) = x(1)$

```py
def ema(arr, wnd):
    alpha = 2 / (wnd + 1)
    res = arr.copy().astype(float)
    for i in range(1, len(arr)):
        res[i] = alpha * res[i] + (1 - alpha) * res[i - 1]
    return res
```

## BOLL（布林带）

布林带有两条，上布林带是移动均值加两倍移动标准差，下布林带是移动均值减两倍移动标准差。

$BOLL(t, n) = SMA(t, n) \pm 2MSD(t, n)$

```py
def msd(arr, wnd):
    res = []
    for i in range(0, len(arr) - wnd + 1):
        res.append(arr[i:i + wnd].std())
    return np.asarray(res)
    
def boll(arr, wnd):
    return sma(arr, wnd) + 2 * msd(arr, wnd), \
           sma(arr, wnd) - 2 * msd(arr, wnd)
```

## ROC（变动率）

也叫简单收益率。其实就是当天的值减去 N 天前的值，反映周期性。

$ROC(t, n) = \frac{x(t) - x(t - n)}{x(t - n)}$

```py
def roc(arr, wnd):
    curr = arr[wnd:]
    prev = arr[:-wnd]
    return (curr - prev) / prev
```

此外，也存在对数收益率的变体。由于对数收益率具有可加性，在一些场合中更加方便。

$\frac{x(t) - x(t-n)}{x(t-n)} \\ = \frac{x(t)}{x(t-n)} - 1 \\ \sim \log(\frac{x(t)}{x(t-n)}) \\ = \log(x(t)) - \log(x(t-n))$

```py
def log_roc(arr, wnd):
    return np.log(arr[wnd:]) - np.log(arr[:-wnd])
```

## RSI（相对强弱指标）

RSI 是 N 天内的涨幅的比例。

RSI 的值总是在 0 到 100 之间，正常情况下在 30 到 70 之间。

$RSI(t, n) = \frac{max(\Delta x(t),0)+\cdots+max(\Delta x(t-n+2),0)}{|\Delta x(t)|+\cdots+|\Delta x(t-n+2)|}$

$\Delta x(t) = x(t) - x(t - 1)$

$RSI(t, n) = \frac{SMA(max(ROC(t, 1), 0), n-1)}{SMA(abs(ROC(t, 1)), n-1)}$

```py
def rsi(arr, wnd):
    roc1 = roc(arr, 1)
    return sma(np.fmax(roc1, 0), wnd - 1) / \
           sma(np.abs(roc1), wnd - 1)
```
