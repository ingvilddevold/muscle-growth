import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def main(input_file: str, output_file: str):

    df = pd.read_csv(input_file)

    weekly_csa = df.groupby(df["t"] // 168)["csa"].last()
    weekly_growth = weekly_csa.diff().dropna()

    plt.figure(figsize=(4, 3))
    plt.plot(range(1, len(weekly_growth)), weekly_growth[:-1], "o-", color="teal")
    plt.xlabel("Week")
    plt.ylabel(r"Net $\Delta A_{CS}$ per week")
    plt.xticks(range(1, len(weekly_growth)), rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)


if __name__ == "__main__":
    app()
