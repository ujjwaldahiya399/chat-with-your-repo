---
title: Chat With Your Repo Backend
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

<!--
When deploying: this file's contents need to become the README.md at
the ROOT of the separate Hugging Face Space git repo (HF Spaces are
their own git repos with their own remote, distinct from this GitHub
repo) — copy manually, not part of this repo's push flow.
-->

# Chat With Your Repo — Backend

FastAPI backend for the Chat with Your Repo RAG project: clones a
GitHub repo, chunks and embeds it, and answers questions with
retrieval-grounded, cited responses via the Groq API.
