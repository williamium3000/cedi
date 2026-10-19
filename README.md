# Contextualized Evaluation of Vision Language Models through Dynamic, Multi-turn Interactions

[![arXiv](https://img.shields.io/badge/arXiv-2607.14499-b31b1b.svg)](https://arxiv.org/abs/2607.14499)

**Paper:** https://arxiv.org/abs/2607.14499 · **Code:** [`main` branch](https://github.com/williamium3000/cedi)

This branch holds the CEDI project page: `index.html` plus `static/` (figures, the
video introduction, and a vendored copy of KaTeX). Serve it with GitHub Pages from
this branch's root.

## Abstract

Multi-modal Large Language Models (MLLMs) have made substantial advances on benchmarks,
yet their real-world effectiveness remains uncertain. This gap stems from the fundamental
misalignment between benchmarks in controlled, static settings and the dynamic,
interactive, and contextualized nature of real-world applications. To bridge this gap, we
propose CEDI (Contextualized Evaluations of MLLMs through Dynamic, multi-round
Interactions), a framework that recasts evaluation as a three-party interaction between an
evaluatee model, an automated examiner, and a grader. The examiner conducts multi-turn,
semi-structured conversation guided by a graph-based representation of the task. By
navigating state-space transitions, CEDI deploys diverse strategies, from clarification
requests to adversarial probes, to elicit performance evidence. We apply CEDI to visual
hallucinations. Empirical results across multiple models, diverse settings, datasets, and
domains show that contextualized, interactive evaluations reveal not only significantly
more hallucinations than conventional static evaluation but also ones that more closely
resemble those arising in practical use cases. We further show that hallucinations often
accumulate over long contexts, through self-reinforcing dialogue history, and models are
particularly vulnerable to questions requiring premise rejection or refusal.
