import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    モデル1 = {"猫": 0.6, "人間": 0.3, "AI": 0.1}
    モデル2 = {"猫": 0.4, "人間": 0.3, "AI": 0.3}
    モデル3 = {"猫": 0.3, "人間": 0.3, "AI": 0.4}
    モデル=[モデル1, モデル2, モデル3]
    def perp(arg):
        for model in モデル:
            p = model[arg]
            l = math.log(p)
            ppl = math.exp(-1/1*l)
            print(f"model={model}, ppl={ppl}")
    perp("猫")
    perp("人間")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
