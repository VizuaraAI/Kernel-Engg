import numpy as np


def blur(arr):
    n = arr.shape[0]
    padded = np.pad(arr, pad_width=1, mode="constant", constant_values=0)
    out = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            out[i, j] = np.mean(padded[i : i + 3, j : j + 3])

    return out


def main():
    arr = np.random.uniform(0, 1, (5, 5)).astype(np.float32)
    blurred = blur(arr)

    print("Input (5x5):")
    print(arr)
    print("\nBlurred (5x5):")
    print(blurred)


if __name__ == "__main__":
    main()

