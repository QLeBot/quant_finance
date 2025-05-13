def calculate_metrics(df):
    total_return = df["returns"].sum()
    sharpe = df["returns"].mean() / df["returns"].std() * (252 ** 0.5)
    return {"total_return": total_return, "sharpe_ratio": sharpe}
