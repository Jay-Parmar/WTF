# WTF - What The File!

**What The File! (WTF)** is a simple tool that lets you chat with your files.

The name is funny. The chatbot isn’t.

Give it a document — PDF, text, code, whatever — and ask questions about it. It’ll try to answer based on the content inside the file. That’s it.

---

## What it does

You provide one or more files. It reads them, builds context, and lets you ask questions like:

- "What’s this file about?"
- "Can you summarize this section?"
- "What does this function do?"
- "What’s the main argument of this paper?"

It’s useful when:
- You’re handed a giant document and don’t want to read all of it.
- You’re debugging or reviewing code you didn’t write.
- You just want a second brain to help you understand content faster.

---

## Features

- Ask questions about the file’s contents
- Works with multiple file types (PDFs, Markdown, code, etc.)
- Answers in plain English
- Still in active development — more features are added regularly

---

## Supported file types (so far)

- `.pdf`
- `.txt`, `.md`

You can give it multiple files too — it tries to build a shared context.

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Jay-Parmar/WTF.git
cd wtf
```

### 2. Create a Virtualenv
```bash
python -m venv venv
```
`And activate it.`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run it
```bash
streamlit run app.py
```

The chatbot will start. You can ask it questions about the file.
