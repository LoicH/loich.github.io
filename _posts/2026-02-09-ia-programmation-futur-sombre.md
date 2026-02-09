---
layout: post
title: [Brouillon] Les IA, la programmation, et un futur sombre
date: 'Mon Feb 9 2026 09:00:00 GMT+0200'
categories: blogging
published: false
---

Pourquoi faire l'effort de réfléchir à la structure d'un programme quand je peux demander à Copilot de générer tout un script à ma place ?

Je vois de plus en plus de gens autour de moi utiliser l'IA générative, dans le graphisme, la musique, mais surtout dans le cadre du développement, et cela me pose problème. Je vais essayer d'expliquer pourquoi.

Cet article est inspiré de la vidéo d'Adam Neely ["Suno, AI Music, and the Bad Future"](https://youtu.be/U8dcFhF0Dlk?si=QM9sNyVVc8VVdoCE). 


# Tout le monde écrit du code par IA

Depuis un moment maintenant, les forums de programmation et de nouvelles technologies sont remplis de posts parlant d'IA, de LLM, de "développement agentique" (https://news.ycombinator.com/, https://www.reddit.com/r/programming/). C'est aussi le cas autour de moi, je rencontre de plus en plus de gens qui utilisent ChatGPT et consorts pour rédiger du code, de la documentation, des présentations, etc.

Et toujours le même constat : les LLMs permettent de produire du code plus rapidement.

Mais à quelle fiabilité ? Je me suis retrouvé à lire des bases de code avec des reliquats de commentaires prouvant que le code autour a été généré par une IA. Je me suis basé sur des pages de documentation qui n'ont pas été rédigées par un humain. Je vois des images génériques pour illustrer des présentations. Ces images n'ont d'ailleurs aucune valeur : elles n'ont aucune cohérence, n'apportent rien de nouveau.

Cette pollution générée par l'IA ("AI slop" en anglais, certains aiment appeler ça du "gloubiboulgIA") a de vraies répercussions : 

- Le créateur d'un logiciel extrêmement utilisé recevait trop de faux signalements de bugs, générés par IA [1]
- L'usage de l'IA a provoqué le licenciement économique de plusieurs personnes dans l'écosystème de l'open source [2]
- Des développeurs en ont tellement marre de lire des contributions générées par IA pour leur projet qu'ils ont développé un système de "réputation" [3]
- L'IA génère des vulnérabilités majeures [4]

# Juste une tendance ?

Est-ce que tout le monde utilise l'IA pour coder ? Qui promeut cette image ? Est-ce que l'IA permet vraiment de coder mieux ou plus vite ? 



# Qui profite de ça ?

- Les techno-capitalistes (Musk, Zuckerberg, etc) qui volent la propriété intellectuelle

- Détruire des livres après les avoir scannés, alors qu'on connaît des méthodes pour scanner sans détruire [5]

# Sources

- [1] "The end of the curl bug-bounty" (https://daniel.haxx.se/blog/2026/01/26/the-end-of-the-curl-bug-bounty/)
- [2] "Vibe Coding Is Killing Open Source Software, Researchers Argue" (https://archive.ph/sgl5M)
- [3] https://github.com/mitchellh/vouch
- [4] "The hottest new vibe coding startup may be a sitting duck for hackers" (https://www.semafor.com/article/05/29/2025/the-hottest-new-vibe-coding-startup-lovable-is-a-sitting-duck-for-hackers)
- [5] " Anthropic destroyed millions of print books to build its AI models " (https://arstechnica.com/ai/2025/06/anthropic-destroyed-millions-of-print-books-to-build-its-ai-models/)

## Vrac

    - deuil de l'artisanat https://news.ycombinator.com/item?id=46926245
    - 

## A lire
- https://news.ycombinator.com/item?id=46933067
https://ezhik.jp/ai-slop-terrifies-me/
- https://localghost.dev/blog/stop-generating-start-thinking/
https://news.ycombinator.com/item?id=46938958
- https://alnrott.medium.com/capitalist-realism-and-the-weaponization-of-bias-in-llms-a-new-paradigm-of-information-control-5a68729e2348
- https://shs.cairn.info/revue-reseaux-2022-1-page-167?lang=fr