import numpy as np


def softmax(x):
    x = np.array(x, dtype=np.float64)

    # Step 1: Numerical stabilization
    x = x - np.max(x)

    # Step 2: Exponential
    exp_x = np.exp(x)

    # Step 3: Normalization
    probabilities = exp_x / np.sum(exp_x)

    return probabilities



# Test cases
test_cases = [
    [2.0, 1.0, 0.1, -1.0],
    [1.0, 2.0, 3.0, 4.0],
    [-4.0, -2.0, 1.0, 5.0],
    [0.0, 0.0, 0.0, 0.0]
]


for i, logits in enumerate(test_cases, 1):

    output = softmax(logits)

    print(f"\nTest Case {i}")
    print("--------------------")

    print("Input:", logits)
    print("Softmax:", output)
    print("Sum:", np.sum(output))
    print("Maximum probability:", np.max(output))
    print("Predicted class:", np.argmax(output))

import csv


with open("test_vectors/softmax_golden.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "test_id",
        "x0",
        "x1",
        "x2",
        "x3",
        "y0",
        "y1",
        "y2",
        "y3"
    ])

    for i, logits in enumerate(test_cases, 1):

        output = softmax(logits)

        writer.writerow([
            i,
            *logits,
            *output
        ])

print("\nGolden test vectors saved to:")
print("test_vectors/softmax_golden.csv")