# Lesson 08 — Parallel Workers with Async

## What changed from Lesson 07

| Lesson 07 | Lesson 08 |
|---|---|
| `anthropic.Anthropic()` | `anthropic.AsyncAnthropic()` |
| `def worker()` | `async def worker()` |
| workers run one by one | `asyncio.gather()` runs all at once |
| time = sum of all workers | time = slowest single worker |

## The one line that matters

```python
results = await asyncio.gather(*[run_one(b) for b in tool_blocks])
```

`gather()` takes any number of coroutines and runs them concurrently.
Results come back in the same order as the inputs, regardless of finish order.

## Sequential vs parallel time

```
Sequential:  [worker1]──[worker2]──[worker3]──  = 3s
Parallel:    [worker1]
             [worker2]                          = 1s (slowest)
             [worker3]
```

## When to use async

Use async when workers are I/O bound — waiting on network, disk, DB.
(All Claude API calls are I/O bound — async is always worth it here.)

Do NOT use async to speed up CPU-bound work — use multiprocessing for that.

## asyncio.gather() vs asyncio.wait()

| | gather() | wait() |
|---|---|---|
| Returns | results in input order | set of completed tasks |
| On error | raises immediately | can handle per-task |
| Use when | you need all results | you want partial results or error control |

For the simple "run all workers, get all results" pattern, `gather()` is correct.
