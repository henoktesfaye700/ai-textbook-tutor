# AI Textbook Tutor: Offline AI Tutor and Learning Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/henoktesfaye700/ai-textbook-tutor/blob/main/ai_textbook_tutor_colab.ipynb)

AI Textbook Tutor is a curriculum-grounded AI tutor for learners with unreliable internet and too few teachers.
Teachers upload the school's textbooks (PDF or EPUB). Students ask questions and receive answers taken only from
those books, with the chapter, the page, a link to the PDF page and a preview of it. Learning progress is stored on the
device and synced to a teacher dashboard whenever a connection is available.

Built with Python, FastAPI, BM25 retrieval, Bayesian knowledge tracing and (optionally) a local language model via llama.cpp.

## Features of this Project

### A. Administrators and Teachers Can

1. Log in to a password-protected dashboard
2. Upload, list, replace and delete textbooks (PDF and EPUB) from the browser
3. Set the subject, grade and language of each book
4. View the number of students, active students, attempts and accuracy
5. Compare results by school, project, class, subject and topic
6. See daily activity, students who need attention and the hardest questions
7. Download results as CSV and school reports as PDF
8. Build and publish signed offline content packs for devices without internet

### B. Students Can

1. Ask questions in plain language and get answers taken only from the textbooks
2. Receive a guiding question before the answer is revealed
3. See the chapter and page of every answer, open the PDF at that page, and preview the page with their words highlighted
4. Filter questions by subject and grade
5. Practise with adaptive quizzes generated from the books
6. Keep working offline once the content is on their device
7. Get AI-written answers in English, Amharic, Afan Oromo or Somali, checked against the textbook (optional)

### C. System Features

1. Refuses questions the textbooks cannot answer instead of guessing
2. Search treats interchangeable Amharic spellings as the same word
3. Removes any sentence in an AI answer that the textbook does not support
4. Anonymous results sync with an optional shared key
5. Optional student access code for sites reachable from the internet
6. Content packs verified with SHA-256 checksums and optional signatures

## How to Install and Run this project?

### Pre-Requisites

1. Install Git Version Control [ https://git-scm.com/ ]
2. Install Python 3.10 or newer [ https://www.python.org/downloads/ ]
3. Install Pip (Package Manager) [ https://pip.pypa.io/en/stable/installing/ ]

### Installation

**1. Clone this project**

```
$ git clone https://github.com/henoktesfaye700/ai-textbook-tutor.git
$ cd ai-textbook-tutor
```

**2. Create a Virtual Environment and Activate**

For Windows

```
$ python -m venv venv
$ venv\Scripts\activate
```

For Mac and Linux

```
$ python3 -m venv venv
$ source venv/bin/activate
```

**3. Install Requirements**

```
$ pip install -r requirements.txt
$ pip install -e .
```

**4. Add your books**

Copy text-based PDF or EPUB files into the `books/` folder (you can select the text with a mouse).
Name each file `subject_g<grade>_<language>.pdf`, for example `biology_g8_en.pdf`, or set the details later in the browser.

**5. Run the Server**

```
$ aitutor serve --admin-password "choose-a-strong-password"
```

| Page | Address | Who |
|---|---|---|
| Student tutor | http://127.0.0.1:8000/ | Students |
| Teacher dashboard | http://127.0.0.1:8000/teacher | Teachers |
| Book library | http://127.0.0.1:8000/teacher/library | Teachers |
| API documentation | http://127.0.0.1:8000/docs | Developers |

**6. Login Credentials**

Log in at `/login` with the password you passed to `--admin-password`.
If you omit the option, a random password is generated and printed once when the server starts.
The dashboard never runs without a password.

## Usage

### Manage books in the browser

Log in, open **Library** and upload a PDF or EPUB with its subject, grade and language. Students can ask about it
immediately. A file with the same name is replaced. Unreadable or scanned files are refused and nothing changes.

### Manage books from the command line

```
$ aitutor import books --out data/passages.jsonl
$ aitutor books
$ aitutor ask "How do plants make food?" --subject biology --grade 8
```

Book details come from `books/manifest.json`, then from the file name pattern, then from defaults:

```json
{"physics.epub": {"subject": "physics", "grade": 7, "lang": "en", "title": "Physics 7"}}
```

### Student devices

```
$ aitutor quiz --subject biology -n 5
$ aitutor sync --url http://SERVER:8000 --class 8A --school "Hawassa Model School" --project "Pilot 2026" --sync-key KEY
```

Quiz results are stored in `outbox.json` and uploaded on the next `aitutor sync`.
A student is counted as one anonymous device in the dashboard.

### Add a language model (optional)

```
$ pip install llama-cpp-python huggingface_hub
$ python -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen2.5-1.5B-Instruct-GGUF','qwen2.5-1.5b-instruct-q4_k_m.gguf',local_dir='models'))"
$ aitutor serve --admin-password "..." --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

The model writes a short answer from the retrieved passages. Every sentence is then checked against those passages,
and unsupported sentences are removed. If nothing is left, the textbook text is shown. Add `--gpu-layers N` to use a GPU.
Check the license of the model you choose.

#### If the model is not available

The tutor works without a language model: it answers with the best-matching textbook passage and its citation.
The model only improves the wording, and it can never take the site down.

| Situation | What happens |
|---|---|
| No `--model` option | Textbook answers. This is the default and needs nothing else. |
| `--model` given, but `llama-cpp-python` is not installed | `aitutor ask` stops with a message and the install command. `aitutor serve` starts without the model, logs a warning, and students get textbook answers. |
| Model file missing or not a valid `.gguf` | The same as above, with a message naming the file. |
| The model fails while answering (for example, out of memory) | That answer falls back to the textbook text, the error is logged, and the student sees no AI label. |
| The model writes sentences the textbook does not support | Those sentences are removed. If none remain, the textbook text is shown. |

### Offline content packs

```
$ aitutor pack build --out school.tutorpack --version 3 --sign-key SECRET
$ aitutor pack install school.tutorpack --sign-key SECRET
$ aitutor serve --pack school.tutorpack --sync-key KEY
$ aitutor pack pull --url http://SERVER:8000 --sync-key KEY --sign-key SECRET
```

A pack contains the passages and the book files. Every file is verified before anything is written, older versions are refused
(use `--force` to override), and a pack with a wrong signature is rejected. Restart `aitutor serve` after installing a pack.

### Google Colab

Upload `ai_textbook_tutor_colab.ipynb` to [Google Colab](https://colab.research.google.com/) and run the cells from top to bottom.
It installs AI Textbook Tutor, imports your books, runs the tutor, quiz and server, adds the optional model, and can create a temporary public link.

### Deploy as a website

Use any host that runs Docker, provides HTTPS and offers a persistent disk.

1. Create a web service from the `Dockerfile`.
2. Set the environment variables `AITUTOR_ADMIN_PASSWORD`, `AITUTOR_STUDENT_CODE`, `AITUTOR_SYNC_KEY` and `AITUTOR_SECRET`.
3. Mount a persistent disk at `/data`.
4. Open `https://your-site/login`, go to **Library**, upload the books, and share `https://your-site/` with the student code.

Only host books you have the right to share.

## Configuration

| Command-line option | Environment variable | Purpose |
|---|---|---|
| `--admin-password` | `AITUTOR_ADMIN_PASSWORD` | Dashboard and library password |
| `--student-code` | `AITUTOR_STUDENT_CODE` | Access code for the student pages |
| `--sync-key` | `AITUTOR_SYNC_KEY` | Key required for result uploads and pack downloads |
| `--pack` | `AITUTOR_PACK` | Content pack published to other devices |
| `--passages`, `--books`, `--db` | `AITUTOR_PASSAGES`, `AITUTOR_BOOKS` | Data locations |
| `--model`, `--gpu-layers` | | Local language model |
| `--host`, `--port` | `PORT` | Network address |
| | `AITUTOR_SECRET` | Signing key for login sessions (keeps logins across restarts) |
| | `AITUTOR_MAX_UPLOAD_MB` | Upload size limit, default 150 |
| | `AITUTOR_FONT` | TrueType font used for Ge'ez text in PDF reports |

## Project Structure

```
src/aitutor/
  api.py                  Server: student page, dashboard, library, sync, logins
  web/                    Student page, teacher dashboard, library, login pages
  importers.py            PDF and EPUB reading
  library.py, cleaning.py Folder import, text cleaning and chunking
  retriever.py            BM25 search with Ge'ez spelling normalisation
  tutor.py                Guiding question, answer and citations
  faithfulness.py         Checks AI answers against the textbook
  bkt.py, quiz.py         Knowledge tracing and adaptive quizzes
  pack.py                 Offline content packs
  preview.py, report.py   Page previews and PDF reports
  cli.py                  The aitutor command
scripts/evaluate.py       Measures search quality on your own questions
tests/                    Automated tests
Dockerfile
ai_textbook_tutor_colab.ipynb          Google Colab notebook
```

## Running the Tests

```
$ pip install -r requirements-dev.txt
$ pytest
```

Measure search quality on your own books with 20 or more labelled questions:

```
$ python scripts/evaluate.py --passages data/passages.jsonl --questions data/eval_questions.jsonl
```

Each line of the questions file looks like `{"q": "How do plants make food?", "topic": "Chapter 1: Photosynthesis"}`.


