from pathlib import Path
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfbase.pdfmetrics import stringWidth


OUT = Path("/Users/keith/Downloads/math_reasoning_competition_presentation_filled.pdf")
PAGESIZE = (13.333 * 72, 7.5 * 72)
W, H = PAGESIZE

BLACK = colors.HexColor("#161616")
GRAY = colors.HexColor("#5F6368")
LIGHT = colors.HexColor("#F3F6F7")
LINE = colors.HexColor("#DCE4E7")
ACCENT = colors.HexColor("#718895")
ACCENT_DARK = colors.HexColor("#33444D")
GREEN = colors.HexColor("#2C8452")
ORANGE = colors.HexColor("#C87936")
RED = colors.HexColor("#A94747")


def txt(c, s, x, y, size=16, color=BLACK, bold=False, align="left"):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if align == "center":
        c.drawCentredString(x, y, s)
    elif align == "right":
        c.drawRightString(x, y, s)
    else:
        c.drawString(x, y, s)


def wrap(s, max_width, size=12, bold=False):
    font = "Helvetica-Bold" if bold else "Helvetica"
    words = s.split()
    lines, cur = [], ""
    for word in words:
        nxt = (cur + " " + word).strip()
        if stringWidth(nxt, font, size) > max_width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = nxt
    if cur:
        lines.append(cur)
    return lines


def para(c, s, x, y, w, size=12, color=BLACK, leading=None, bold=False):
    leading = leading or size * 1.35
    for line in wrap(s, w, size=size, bold=bold):
        txt(c, line, x, y, size=size, color=color, bold=bold)
        y -= leading
    return y


def bullets(c, items, x, y, w, size=12, color=BLACK, leading=None):
    leading = leading or size * 1.45
    for item in items:
        lines = wrap(item, w - 18, size=size)
        if not lines:
            continue
        txt(c, "•", x, y, size=size, color=color)
        txt(c, lines[0], x + 14, y, size=size, color=color)
        y -= leading
        for line in lines[1:]:
            txt(c, line, x + 14, y, size=size, color=color)
            y -= leading
        y -= 2
    return y


def title(c, s, subtitle=None):
    txt(c, s, 48, H - 58, size=21, bold=False)
    if subtitle:
        txt(c, subtitle, 48, H - 84, size=10.5, color=GRAY)


def page_num(c, n):
    txt(c, str(n), W - 34, 20, size=8, color=colors.HexColor("#A0A0A0"), align="right")


def section(c, name):
    txt(c, name, W / 2, H / 2, size=32, align="center")


def card(c, x, y, w, h, fill=LIGHT, stroke=LINE):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.roundRect(x, y, w, h, 10, stroke=1, fill=1)


def simple_table(c, headers, rows, x, y, widths, row_h=28, size=9.5):
    total_w = sum(widths)
    c.setStrokeColor(LINE)
    c.setFillColor(ACCENT_DARK)
    c.rect(x, y - row_h, total_w, row_h, stroke=0, fill=1)
    cx = x
    for h, ww in zip(headers, widths):
        txt(c, h, cx + 7, y - 18, size=size, color=colors.white, bold=True)
        cx += ww
    yy = y - row_h
    for i, row in enumerate(rows):
        yy -= row_h
        c.setFillColor(colors.HexColor("#F8FAFB") if i % 2 == 0 else colors.HexColor("#EDF3F5"))
        c.rect(x, yy, total_w, row_h, stroke=0, fill=1)
        cx = x
        for val, ww in zip(row, widths):
            para(c, str(val), cx + 7, yy + row_h - 12, ww - 12, size=size, leading=size * 1.05)
            cx += ww
    c.setStrokeColor(LINE)
    c.rect(x, yy, total_w, (len(rows) + 1) * row_h, stroke=1, fill=0)


def bar(c, labels, values, x, y, w, h, maxv=100):
    gap = 18
    bw = (w - gap * (len(values) - 1)) / len(values)
    palette = [ACCENT, ORANGE, GREEN, ACCENT_DARK]
    for i, (label, val) in enumerate(zip(labels, values)):
        bx = x + i * (bw + gap)
        bh = h * val / maxv
        c.setFillColor(palette[i % len(palette)])
        c.rect(bx, y, bw, bh, stroke=0, fill=1)
        txt(c, f"{val:.1f}%", bx + bw / 2, y + bh + 10, size=10, bold=True, align="center")
        for j, line in enumerate(label.split("\n")):
            txt(c, line, bx + bw / 2, y - 15 - j * 11, size=8.5, color=GRAY, align="center")


def flow(c, steps, x, y, w, h=50):
    gap = 11
    step_w = (w - gap * (len(steps) - 1)) / len(steps)
    inner = step_w - 8
    # Pick one font size so the widest single word fits every card (no clipping).
    longest = max(stringWidth(word, "Helvetica-Bold", 10) for s in steps for word in s.split())
    size = 9.0
    while size > 6.0 and longest * (size / 10.0) > inner:
        size -= 0.25
    for i, step in enumerate(steps):
        sx = x + i * (step_w + gap)
        card(c, sx, y, step_w, h, fill=colors.HexColor("#F7FAFB"))
        lines = wrap(step, inner, size=size, bold=True)
        leading = size * 1.18
        ty = y + h / 2 + (len(lines) - 1) * leading / 2 + size * 0.33
        for line in lines:
            txt(c, line, sx + step_w / 2, ty, size=size, color=ACCENT_DARK, bold=True, align="center")
            ty -= leading
        if i < len(steps) - 1:
            txt(c, "→", sx + step_w + gap / 2, y + h / 2 - 4, size=9, color=GRAY, align="center")


slides = []


def add_slide(fn):
    slides.append(fn)
    return fn


@add_slide
def s1(c):
    txt(c, "Improving Math Reasoning Under a Token Budget", W / 2, H * 0.57, 28, align="center")
    txt(c, "CSE 151B/251B Math Reasoning Competition - Keith Gong", W / 2, H * 0.51, 15, GRAY, align="center")
    txt(c, "Qwen3-4B-Thinking | vLLM | FRQ Repair | Kaggle Submission", W / 2, H * 0.43, 11, ACCENT_DARK, align="center")


@add_slide
def s2(c):
    title(c, "Summary")
    y = H - 108
    body = [
        "This project solves the CSE 151B/251B math reasoning competition using Qwen3-4B-Thinking served through vLLM.",
        "The task contains both multiple-choice and free-response math problems, which require different extraction and formatting behavior.",
        "Early experiments showed that generic changes such as self-consistency and LoRA often hurt because they reduced the model's effective reasoning budget.",
        "The final approach keeps MCQ close to the strong baseline and improves FRQ through targeted postprocessing, template repair, and submission validation."
    ]
    y = bullets(c, body, 70, y, 560, size=12.4)
    card(c, 665, H - 280, 210, 170)
    txt(c, "Kaggle public", 770, H - 145, 12, GRAY, align="center")
    txt(c, "0.650 → 0.671", 770, H - 182, 25, GREEN, bold=True, align="center")
    txt(c, "FRQ public repair", 770, H - 228, 12, GRAY, align="center")
    txt(c, "82.56%", 770, H - 260, 22, ACCENT_DARK, bold=True, align="center")
    bullets(c, [
        "Team: Keith Gong",
        "Main method: base model inference + deterministic FRQ repair",
        "Key lesson: answer extraction and token budget are modeling constraints"
    ], 72, 105, 760, size=12.5, color=ACCENT_DARK)


@add_slide
def s3(c):
    title(c, "Key Words")
    labels = [
        ("Qwen3\nThinking\nModel", 355, 260, 78, ACCENT),
        ("vLLM", 495, 280, 42, BLACK),
        ("FRQ\nRepair", 610, 250, 62, ACCENT_DARK),
        ("Token\nBudget", 735, 280, 48, ACCENT),
        ("Kaggle", 240, 288, 45, BLACK),
    ]
    c.setStrokeColor(colors.HexColor("#C8D1D5"))
    c.line(205, 310, 790, 310)
    for text, x, y, r, color in labels:
        c.setFillColor(color)
        c.setStrokeColor(color)
        c.circle(x, y, r, stroke=1, fill=1)
        for i, line in enumerate(text.split("\n")):
            txt(c, line, x, y + (len(text.split("\n")) - 1) * 8 - i * 16 - 4, size=12, color=colors.white, align="center")
    para(c, "The project became less about finding a bigger model and more about controlling how a small model reasons, formats, and survives the grader.", 170, 95, 610, size=12, color=GRAY)


@add_slide
def s4(c):
    section(c, "Introduction")


@add_slide
def s5(c):
    title(c, "Team Introduction")
    txt(c, "Keith Gong", 68, H - 140, 24, bold=True)
    para(c, "Department of Data Science, University of California, San Diego\nx4gong@ucsd.edu", 70, H - 175, 350, size=12.5, color=GRAY)
    card(c, 500, H - 345, 360, 220)
    txt(c, "Project role", 525, H - 160, 17, bold=True)
    bullets(c, [
        "Built and debugged the vLLM inference pipeline.",
        "Ran remote GPU experiments on RunPod/DataHub.",
        "Designed MCQ/FRQ-specific prompts and extraction logic.",
        "Implemented scoring, repair, CSV validation, and submission assembly.",
        "Compared prompting, postprocessing, LoRA, and template-repair strategies."
    ], 525, H - 195, 300, size=11.2)


@add_slide
def s6(c):
    section(c, "Methodology")


@add_slide
def s7(c):
    title(c, "Data Processing")
    para(c, "The split has different answer contracts, so preprocessing keeps MCQ and FRQ separate instead of treating them as one generic generation task.", 68, H - 105, 720, size=12.2, color=GRAY)
    simple_table(c, ["Dataset", "Rows", "Use"], [
        ["Public", "1126", "Evaluation and controlled error analysis"],
        ["Public MCQ", "375", "Baseline prompt and letter extraction"],
        ["Public FRQ", "751", "Prompting, repair, template analysis"],
        ["Private", "943", "Submission generation only"],
        ["Private MCQ / FRQ", "300 / 643", "Merged into final Kaggle CSV"],
    ], 70, H - 150, [140, 80, 330], row_h=29, size=9.5)
    txt(c, "Pipeline", 747, H - 150, 16, bold=True, align="center")
    flow(c, ["JSONL input", "detect options", "type prompt", "vLLM", "boxed extraction", "repair / validate", "CSV"], 560, H - 272, 375)
    bullets(c, [
        "Public answers were used to measure and categorize errors.",
        "Private predictions kept full model traces, matching the required submission format.",
        "Every CSV was checked for row count, unique ids, blanks, boxed answers, and valid MCQ letters."
    ], 600, H - 330, 290, size=10.5)


@add_slide
def s8(c):
    title(c, "Deep Learning Model")
    txt(c, "Qwen3-4B-Thinking-2507", 70, H - 135, 24, bold=True)
    para(c, "The base model is a small thinking model: before returning a final answer, it often emits a long reasoning trace. That trace improves math performance, but it also makes inference slow and sensitive to max token limits.", 72, H - 172, 460, size=12.2)
    simple_table(c, ["Component", "Final decision"], [
        ["Inference", "vLLM on one A40 GPU"],
        ["Weights", "BitsAndBytes quantization"],
        ["Sampling", "T=0.6, top-p=0.95, top-k=20"],
        ["MCQ", "Keep baseline behavior"],
        ["FRQ", "Base generation + deterministic repair"],
    ], 545, H - 135, [120, 245], row_h=30, size=9.5)
    card(c, 70, 82, 800, 80, fill=colors.HexColor("#F8F5EF"), stroke=colors.HexColor("#E6D8C5"))
    para(c, "Key finding: the model was not simply unable to reason. A large share of failures came from truncation, malformed final answers, answer-count mistakes, or extractor-grader mismatch.", 95, 132, 750, size=12.2, color=ACCENT_DARK)


@add_slide
def s9(c):
    title(c, "Engineering Tricks")
    labels = [
        ("Remote execution", ["nohup for long jobs", "PID files and tail -f logs", "nvidia-smi for GPU checks"]),
        ("Memory and speed", ["vLLM made full runs practical", "High-token runs could take 8+ hours", "KV cache pressure controlled generation settings"]),
        ("Submission validation", ["row count and duplicate checks", "blank and boxed-answer checks", "valid MCQ final-letter checks"]),
    ]
    for i, (head, items) in enumerate(labels):
        x = 70 + i * 290
        card(c, x, H - 335, 245, 210)
        txt(c, head, x + 18, H - 160, 15, bold=True, color=ACCENT_DARK)
        bullets(c, items, x + 20, H - 198, 200, size=10.8)
    para(c, "Rule of thumb: anything over five minutes ran remotely with logs; notebooks were reserved for final reproducibility and Kaggle packaging.", 120, 110, 720, size=13, color=ACCENT_DARK)


@add_slide
def s10(c):
    section(c, "Experiments")


@add_slide
def s11(c):
    title(c, "Experiment 1", "Token budget, baseline, and repair")
    txt(c, "Milestone 100-question subset", 80, H - 130, 13, color=GRAY)
    bar(c, ["MCQ\nbaseline", "FRQ\nbaseline", "FRQ\nstrict"], [71.1, 54.8, 56.5], 105, H - 355, 280, 170)
    txt(c, "Full public FRQ pipeline", 545, H - 130, 13, color=GRAY)
    bar(c, ["Raw", "Baseline\nrepair", "Final\nrepair"], [56.6, 77.1, 82.6], 570, H - 355, 280, 170)
    bullets(c, [
        "Reducing max_tokens made experiments faster but hurt MCQ because the model often had not reached its final boxed letter.",
        "FRQ was more recoverable: many wrong rows contained usable math but malformed final answers.",
        "Final public FRQ repair reached 620/751 = 82.56%, with an oracle ceiling of 83.62% for the current candidate set."
    ], 82, 120, 795, size=11.8)


@add_slide
def s12(c):
    title(c, "Experiment 2", "LoRA/SFT did not beat controlled repair")
    simple_table(c, ["Attempt", "Training target", "Observed outcome"], [
        ["Public FRQ LoRA", "751 public FRQ answers", "Overfit behavior; worse sampled eval"],
        ["OpenR1 LoRA", "External solution traces", "Solution style mismatch; too verbose"],
        ["Self-distill LoRA", "Base-correct public FRQ", "Holdout repaired: 110/200 = 55.0%"],
        ["Final decision", "No adapter", "Base model + repair beat adapters"],
    ], 70, H - 130, [150, 260, 370], row_h=39, size=9.5)
    card(c, 95, 88, 760, 78, fill=colors.HexColor("#F8F5EF"), stroke=colors.HexColor("#E6D8C5"))
    txt(c, "Takeaway", 120, 134, 12, color=ORANGE, bold=True)
    para(c, "Fine-tuning only helps if the training target matches the grader exactly. Otherwise it teaches the model to be verbose, not correct.", 205, 140, 610, size=12, color=BLACK)


@add_slide
def s13(c):
    section(c, "Discussion")


@add_slide
def s14(c):
    title(c, "Strength/Weakness/Improvement")
    cols = [
        ("3 Strengths", GREEN, [
            "Separated MCQ and FRQ instead of forcing one prompt to solve both.",
            "Used the course judger to measure every repair candidate.",
            "Validated CSV format before submission, catching invalid boxes and labels."
        ]),
        ("3 Weaknesses", RED, [
            "Some template repairs can overfit public-style problem patterns.",
            "Full private generation is slow and GPU-memory sensitive.",
            "LoRA attempts did not generalize under the time budget."
        ]),
        ("Improvements", ACCENT_DARK, [
            "Build private-safe topic solvers rather than public-specific patches.",
            "Use selective second-pass generation on only uncertain FRQ.",
            "Collect external data in the exact final-answer style before retrying SFT."
        ]),
    ]
    for i, (head, color, items) in enumerate(cols):
        x = 70 + i * 290
        txt(c, head, x, H - 130, 16, color=color, bold=True)
        bullets(c, items, x, H - 170, 240, size=11.5)


@add_slide
def s15(c):
    title(c, "What have you learned")
    bullets(c, [
        "The reasoning trace is not extra decoration; for Qwen3-4B-Thinking it is where much of the accuracy comes from.",
        "MCQ and FRQ have different failure modes: MCQ needs a valid final letter, while FRQ often needs symbolic cleanup.",
        "Postprocessing is a measurable part of the model system when the grader checks exact symbolic answers.",
        "Fine-tuning can hurt when it changes the model's answer style without improving the exact final answer.",
        "The safest leaderboard gains came from analyzing actual error rows instead of guessing."
    ], 75, H - 125, 610, size=12.6)
    card(c, 720, H - 295, 160, 170)
    txt(c, "Score movement", 800, H - 150, 11, color=GRAY, align="center")
    txt(c, "0.650", 800, H - 185, 20, color=GRAY, align="center")
    txt(c, "↓", 800, H - 220, 16, color=GRAY, align="center")
    txt(c, "0.671", 800, H - 252, 25, color=GREEN, bold=True, align="center")


@add_slide
def s16(c):
    title(c, "Future Work")
    bullets(c, [
        "Selective second-pass generation for wrong or low-confidence FRQ only, instead of regenerating the full private set.",
        "A private-safe repair library organized by topic: statistics, exponential growth, trigonometry, algebra, and expression formatting.",
        "Better answer-count detection for multi-blank free-response questions.",
        "External data filtered to competition-like final-answer format before trying LoRA again.",
        "Compare thinking and non-thinking Qwen models under the same wall-clock budget.",
        "A fully self-contained Kaggle notebook that always produces submission.csv as an output artifact."
    ], 75, H - 118, 610, size=12.1)
    card(c, 720, H - 295, 165, 170)
    txt(c, "Next target", 802, H - 155, 11, GRAY, align="center")
    txt(c, "85%+", 802, H - 202, 30, ACCENT_DARK, bold=True, align="center")
    txt(c, "FRQ accuracy", 802, H - 230, 11, GRAY, align="center")
    para(c, "with less public-set leakage and stronger topic generalization", 745, H - 260, 115, size=9, color=GRAY)


c = canvas.Canvas(str(OUT), pagesize=PAGESIZE)
for i, fn in enumerate(slides, 1):
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    fn(c)
    page_num(c, i)
    c.showPage()
c.save()
print(OUT)
