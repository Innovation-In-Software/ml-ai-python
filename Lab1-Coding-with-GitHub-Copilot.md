# Lab 1: Coding with GitHub Copilot

> **Machine Learning with AI and Python** · Day 1

## The scenario

You have just joined a data team. Your first assignment is small but real: build a tested Python utility for reading and summarizing incoming data, using GitHub Copilot the way the team does. By the end you will have a working `data_utils.py` module with tests, and, more importantly, the habits that separate people who *use* an AI assistant from people who get burned by one.

## Why this lab matters

Working developers now write code next to an AI assistant. Copilot is fast, and it is confidently wrong often enough that the real skill is steering it and verifying its output. This lab practices every Copilot surface from the slides (inline completions, the five-step workflow, Ask/Edit/Agent modes, `/explain`, `/fix`, `/tests`) and adds the professional habits around them: writing intent as context, reviewing diffs, refactoring, and never trusting code you have not run.

## What you will do

- Part A: Drive inline completions and understand where suggestions come from
- Part B: Use Copilot Chat to explain and repair code
- Part C: Guide Copilot with comments and docstrings (intent as a spec)
- Part D: Test everything, and catch a wrong suggestion
- Part E: Use Ask, Edit, and Agent modes, and inline chat, for bigger changes
- Part F (capstone): Build and test a small `data_utils.py` module end to end

## Before you start

You will work in VS Code with GitHub Copilot and Python. Open VS Code, create a folder for today (for example `ml-day1`), and open it. Confirm the Copilot icon in the status bar shows it is active.

> [!NOTE]
> If the Copilot icon shows it is not active, click it and confirm you are signed in to GitHub. Ask your instructor if it still will not connect.

---

## Part A: Get productive with inline completions

### Step 1: Turn a comment into code

**Why:** The fastest way to steer Copilot is to state your intent in a plain comment, then let it draft the code. This is the everyday inline-completion workflow, and it is where you will spend most of your time.

1. Create a file named `copilot_play.py`.
2. Type the comment and the function header below, then pause. Copilot proposes the body as gray "ghost text."
3. Press `Tab` to accept, `Esc` to dismiss, and `Alt+]` or `Alt+[` to cycle through alternatives.

```python
# Return the average of a list of numbers, or 0.0 if the list is empty
def average(numbers):
    # let Copilot suggest the body, then press Tab to accept
```

4. Add these checks and run the file (right-click, then "Run Python File", or run `python copilot_play.py`):

```python
print(average([2, 4, 6]))   # 4.0
print(average([]))          # 0.0
```

> [!NOTE]
> **Checkpoint:** You accepted a suggestion and `average([])` returns `0.0` instead of crashing. If it crashed, that is exactly why we verify. Fix it in Step 5's style, or ask Copilot to handle the empty case.

### Step 2: See where context comes from

**Why:** Copilot does not read your mind. It reads your open files, the code near your cursor, your comments, and your names. Better context produces better suggestions. Proving this to yourself changes how you write.

1. At the top of `copilot_play.py`, add a comment describing the file's purpose:

```python
# Utilities for reading and summarizing incoming home-sales data.
```

2. Now start a new function with a vague name and watch the suggestion, then rename it to something clear and watch it improve:

```python
def f(x):          # vague: weak suggestions
    ...

def median_price(prices):   # clear: Copilot infers the intent
    ...
```

> [!TIP]
> Two levers control quality: the comment right above the cursor, and the names you choose. Clear names are not just style; they are context the model uses.

---

## Part B: Explain and repair with Copilot Chat

### Step 3: Ask Copilot to explain code

**Why:** `/explain` is the fastest way to understand code you did not write. Reach for it before you change unfamiliar code, so you do not break something you never understood.

1. Open Copilot Chat (the chat icon in the sidebar, or `Ctrl+Alt+I`).
2. Select the `average` function.
3. Type `/explain` and read the answer. Does it match what you intended?
4. Follow up in plain English: "What happens if `numbers` contains a string?" Read the answer critically.

### Step 4: Break something, then use /fix

**Why:** Copilot can propose a targeted patch for an error, shown as a diff you approve. You stay in control because you review the change, not just the result.

1. Paste this function. It crashes on a zero divisor.

```python
# this crashes when b is 0
def safe_divide(a, b):
    return a / b

print(safe_divide(10, 0))   # ZeroDivisionError until you /fix it
```

2. Select the function, open Chat, and type `/fix`.
3. Review the proposed diff. It should guard `b == 0` or catch the exception. Accept it and run again.

> [!NOTE]
> **Checkpoint:** The program no longer crashes on `safe_divide(10, 0)`, and you can say in one sentence *what* changed.

---

## Part C: Guide Copilot with intent

### Step 5: Write a docstring as a spec

**Why:** A docstring is a strong intent signal. Copilot reads it and generates a body that matches the described behavior, which is more reliable than a one-line comment. Writing the spec first is also good engineering.

1. Type this and let Copilot fill the body from the docstring:

```python
def to_fahrenheit(celsius):
    """Convert a temperature in Celsius to Fahrenheit and return it as a float.

    Raise ValueError if celsius is below absolute zero (-273.15).
    """
    # let Copilot use the docstring to write the body
```

2. Test it:

```python
print(to_fahrenheit(0))     # 32.0
print(to_fahrenheit(100))   # 212.0
# to_fahrenheit(-300)       # should raise ValueError
```

> [!NOTE]
> **Checkpoint:** The docstring alone produced a correct body, and the two known values match. Confirm the ValueError path exists.

### Step 6: Comment-driven parsing

**Why:** Real input is messy. Describing the messiness in a comment lets Copilot handle it, and gives you a concrete case to verify.

1. Write the intent, then let Copilot draft it:

```python
# Parse a price like "$450,000" or "450000" into a float.
# Return None if the text is not a valid number.
def parse_price(text):
    ...
```

2. Verify against known inputs:

```python
print(parse_price("$450,000"))   # 450000.0
print(parse_price("450000"))     # 450000.0
print(parse_price("N/A"))        # None
```

---

## Part D: Test everything, and catch a wrong suggestion

### Step 7: Generate tests with /tests

**Why:** Tests are how you prove code does what you meant. Copilot scaffolds them, including edge cases, so you spend your time judging tests instead of typing boilerplate.

1. Select `to_fahrenheit`, open Chat, and type `/tests`.
2. Skim the generated tests. Do they cover the freezing point, boiling point, and the ValueError? Add any that are missing.
3. Run the tests (the generated file usually runs with `python -m pytest` or as a script).

### Step 8: Catch a wrong suggestion by testing it

**Why:** "It ran" is not "it is correct." Copilot is subtly wrong often enough that testing against known answers is your safety net.

1. Ask Copilot to "write a function `is_prime(n)`."
2. Do not trust it on sight. Run it against known answers:

```python
for n, expected in [(1, False), (2, True), (9, False), (13, True), (1_000_003, True)]:
    got = is_prime(n)
    print(n, got, "OK" if got == expected else "WRONG")
```

3. If any line prints `WRONG`, fix it (try `/fix`, or correct it yourself) and rerun until all pass.

> [!NOTE]
> **Checkpoint:** You found and fixed at least one issue by testing, not by reading. Common bugs: treating `1` as prime, mishandling `2`, or being very slow on large inputs.

---

## Part E: Modes for bigger changes

### Step 9: Edit in place with inline chat

**Why:** Inline chat changes selected code right where it is. Use it for small, local edits without leaving the editor.

1. Select `parse_price`.
2. Press `Ctrl+I` (`Cmd+I` on macOS).
3. Type: "add type hints and a short docstring." Review the diff and accept or reject.

### Step 10: Ask, Edit, and Agent modes

**Why:** Chat modes trade control for automation. Knowing which mode you are in keeps you from being surprised by changes.

Switch modes at the bottom of the Chat input box and try each:

- **Ask** (advice only): "How would you make `average` reject non-numeric values?" Copilot explains; you make the change.
- **Edit** (edits the files you point it at): ask it to "add input validation to `average` in `copilot_play.py`." Review the diff.
- **Agent** (plans and runs multi-step work): "Add a `median(numbers)` function with a docstring and tests." Approve each step.

> [!WARNING]
> In Agent mode, read every proposed command and file change before you approve it. You stay responsible for what runs on your machine, including any terminal command it suggests.

> [!NOTE]
> **Checkpoint:** You have used all three modes and can state, in one sentence each, when you would reach for Ask, Edit, or Agent.

---

## Part F (capstone): Build a tested `data_utils.py` module

**Why:** Real work is not single functions, it is small, tested modules. Here you drive Copilot end to end to build one, then prove it works. This is the exact loop you will repeat all week.

### Step 11: Write the spec

Create `data_utils.py`. Paste this module docstring and the two function stubs, and let Copilot implement each from its docstring:

```python
"""Utilities for cleaning and summarizing incoming numeric data."""

def clean_number(value):
    """Convert value to a float. Strip a leading '$' and any commas.
    Return None if it cannot be converted (for example "N/A" or None).
    """
    ...

def summarize(numbers):
    """Return a dict with count, min, max, mean, and median of a list of
    numbers. Ignore None values. For an empty list, return counts of 0 and
    None for the statistics.
    """
    ...
```

### Step 12: Prove it works

Create `test_data_utils.py` (or ask Copilot with `/tests`), then verify against these known answers:

```python
from data_utils import clean_number, summarize

rows = ["$1,200", "3400", "N/A", None, "2,000"]
nums = [clean_number(r) for r in rows]
print(nums)                     # [1200.0, 3400.0, None, None, 2000.0]

clean = [n for n in nums if n is not None]
print(summarize(clean))
# {'count': 3, 'min': 1200.0, 'max': 3400.0, 'mean': 2200.0, 'median': 2000.0}

print(summarize([]))            # count 0, statistics None
```

> [!NOTE]
> **Checkpoint:** Both printouts match, including the empty-list case. If they do not, use `/fix` and rerun. You now have a small, tested module you built with an AI pair programmer and verified yourself.

### Step 13: Document it

Ask Copilot Chat: "Write a short README section documenting `clean_number` and `summarize`, with one usage example each." Read it, correct anything inaccurate, and save it as `README.md`.

> [!TIP]
> Copilot is good at first-draft documentation, but it will sometimes describe behavior your code does not have. Treat generated docs like generated code: verify against what the code actually does.

---

## On your own

Extend `data_utils.py` with one function your team would actually want, for example `parse_percent("12.5%") -> 0.125`, or `to_title_case(" reno ") -> "Reno"`. Build it with a docstring-first prompt, generate tests with `/tests`, and add at least one known-answer check of your own. Remove anything sensitive before you type it into Copilot.

## Responsible AI

> [!IMPORTANT]
> - Copilot is a cloud service: your open files and prompts are sent to it. Never paste secrets, API keys, or private data into your code or chat.
> - Read and run generated code before you trust it. Running without errors is not the same as being correct.
> - You own every line you keep, including code an agent wrote and commands it ran.

## What you learned

- Steer Copilot with comments, docstrings, and clear names, and accept or reject suggestions deliberately
- `/explain`, `/fix`, and `/tests` cover understanding, repairing, and testing code
- Inline chat handles quick local edits; Ask, Edit, and Agent modes trade control for automation
- The professional loop: state intent, review the diff, run the code, and verify against known answers

## Common Copilot pitfalls to avoid

- Accepting a suggestion because it looks plausible, without running it
- Letting Agent mode run terminal commands you did not read
- Vague prompts and vague names, then blaming the tool for vague output
- Trusting generated documentation or comments that describe behavior the code does not have
