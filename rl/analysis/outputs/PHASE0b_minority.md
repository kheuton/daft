# Phase 0b — true-minority band characterization

## Depth of the correct answer in the true-minority band

C = total correct samples (of 64). rank = position of best correct class when classes are sorted by vote count (2 = correct is runner-up).

| arm | minority probs | mean C | C=1 | C 2-3 | C 4-7 | C 8-15 | C 16+ | correct is rank-2 | mean best_wrong |
|---|---|---|---|---|---|---|---|---|---|
| pi0 | 532 | 4.8 | 21% | 25% | 34% | 19% | 1% | 19% | 15.0 |
| grpo | 546 | 4.7 | 23% | 26% | 31% | 16% | 3% | 18% | 14.6 |
| passk | 559 | 4.8 | 19% | 29% | 32% | 17% | 3% | 18% | 14.8 |
| votek | 548 | 4.9 | 23% | 26% | 29% | 21% | 2% | 22% | 14.6 |

## Is the confidently-wrong mode stable across arms?

- true-minority in pi0: 532; in grpo: 546; in BOTH: 399 (75% of the smaller)
- of the shared 399, the dominant WRONG canonical is IDENTICAL pi0 vs grpo in 262 (66%) => the wrong mode is a stable, systematic error, not sampling noise.

- true-minority in ALL 4 arms: 306 problems (persist through RL of every kind).

## Concrete examples (grpo): correct answer vs the winning wrong answer

| problem_id | gt answer | dominant WRONG canonical | wrong votes | correct votes (C) | correct rank |
|---|---|---|---|---|---|
| What is the greatest multiple of 99 that is less than 0? | -99 | 0 | 30 | 26 | 2 |
| A line segment of length $5$ has one endpoint at $(1, 2)$ and the other endpoint | 6, -2 | -2,6 | 27 | 24 | 2 |
| Find the remainder when $1 + 2 + 2^2 + 2^3 + \dots + 2^{100}$ is divided by 7. | 3 | 1 | 26 | 24 | 2 |
| Define \[f(x) = (x-1)(x-3)(x-7)(x-9).\]Evaluate $f(6) - f(4)$. | 0 | 90 | 25 | 23 | 2 |
| Let $x$ be a real number such that $\sec x - \tan x = 2.$  Find $\sec x + \tan x | \frac{1}{2} | 2 | 23 | 22 | 2 |
| In a class of $30$ students, exactly $7$ have been to Mexico and exactly $11$ ha | 16 | 12 | 31 | 21 | 2 |
| A chocolate chip cookie recipe calls for 15 cups of flour for 20 dozen cookies.  | 9 | 108 | 23 | 20 | 2 |
| Suppose that $g(x)=f^{-1}(x)$. If $g(-15)=0$, $g(0)=3$, $g(3)=9$ and $g(9)=20$,  | 0 | 9 | 21 | 19 | 2 |
| If $f(a) = \frac{1}{1-a}$, find the product $f^{-1}(a) \times a \times f(a)$.  ( | -1 | 1 | 20 | 19 | 2 |
| Solve for $x$, where $x > 0$ and $0 = -21x^2 - 11x + 40.$ Express your answer as | \dfrac{8}{7} | \frac{10}{21} | 20 | 18 | 2 |
| Let $f(x) = \sqrt{x}$ and $g(x) = x^2.$ Find $f(g(f(g(f(8))))).$ | 2\sqrt{2} | 8 | 24 | 18 | 2 |
| test/algebra/2102.json | 56 | 80 | 18 | 17 | 2 |
| Let $n$ be the inverse of $2\pmod{17}$. That is, let $n$ be the integer $0\leq n | 2 | 16 | 18 | 17 | 2 |
| test/intermediate_algebra/102.json | 2 | 1 | 25 | 17 | 2 |
| The sequence $(a_n)$ is defined by $a_1 = 1,$ $a_2 = 2,$ and
\[a_n^2 - a_{n - 1} | 100 | 1 | 22 | 16 | 2 |
| Five people can mow a lawn in 12 hours.  How many more people are needed to mow  | 15 | 20 | 23 | 15 | 2 |
| test/intermediate_algebra/1000.json | 3 | 0 | 16 | 15 | 2 |
| Let $a,$ $b,$ and $c$ be distinct complex numbers such that
\begin{align*}
a^3 & | 15 | 2 | 18 | 15 | 2 |

