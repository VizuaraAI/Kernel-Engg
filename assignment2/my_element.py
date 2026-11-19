"""Your own element of C, for Problems 1, 7 and 13. Runs on your laptop; no GPU, nothing is sent anywhere.

    python my_element.py you@example.com

Use the email you registered for the workshop with. The element is C[r][c] of the M = 1,024 product (C is
1,024 x 12,288), and it is different for almost every student, so your answers to Problems 1, 7 and 13 are your own.
"""
import hashlib
import sys

if len(sys.argv) != 2 or "@" not in sys.argv[1]:
    sys.exit("usage: python my_element.py you@example.com")
h = hashlib.sha256(sys.argv[1].strip().lower().encode()).hexdigest()
r, c = int(h[:8], 16) % 1024, int(h[8:16], 16) % 12288
print(f"Your element is C[{r}][{c}] of the M = 1,024 product: row {r} of 1,024, column {c} of 12,288.")
print(f"It is row {r} of X (the token) times column {c} of W, 4,096 multiply-adds.")
