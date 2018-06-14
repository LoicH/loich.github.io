---
layout: post
title: Ad targetting with reinforcement learning
date: 'Sun Jun 10 2018 02:00:00 GMT+0200'
categories: datascience
published: false
---
## Intro

Data science is used for lots of tasks, including selecting the best ads to make users click on it, because companies need people to buy their product!

In this article I will explain **how to perform ad targetting** with reinforcement learning.

## Our goal

We are managing a news website with lots of articles and pages, visited by many users. We want to monetize our content by placing ads on our website, but we can't randomly assign ads to articles, we need to find a way to put relevant ads in in articles. 
Let's say everytime a user requests page we can choose which ad the user will see, and we can record whether he clicked on the ad.

How can we maximize the number of clicks on our ads?

## Multi-armed bandits

This is a multi-armed bandit problem: at each turn we can choose one action from a limited choice, without knowledge of the reward associated to each action. We can only learn from past experiences what actions are the most lucrative, to choose the future actions.

As an example, let's assume we are in a waird casino with some free bandit machines in front of us. These machinese do not require us to put money inside, we juste have to push a button and a reward can come. We have no knowledge about these machines, there may be one machine that will give us 1$ every time we press a button, and another machine that give us 100$ once every 10 pushes, and nothing 9 times out of 10. There may also be a machine that give us 2$ once every 100 push and nothing 99% percent of the time... We want to be as rich as possible before the casino owner kicks us out!

How can we do this? You can try to follow your intuition, but it is not very interesting. For instance let's say we have the 3 previous machines in front of us, with the following characteristics:
- Machine #1: 1$ every time
- Machine #2: 100$ 1 time out of 10
- Machine #3: 2$ 1 time out of 100.

If we knew the odds beforehand, we should frantically push on the 2nd machine's button, which has the best mean outcome, approx. 10$ for every push of the button. But we don't know that. We only have 3 similar-looking machines in front of us, and so little time to drain the money out of them!

Let's try some scenarios:
### Scenario #1
- You try the first machine, 

Fortunately there is an optimal way of choosing the actions which is a compromise between selecting the best action we found so far, and exploring to find better actions.

## Evaluation

## Let's start with the data

you already have a website with some different articles, and some various ads on it. You collected data on which ads were clicked on which page, and you get a file a bit like this:

````
0:0.74;0.83;0.07;0.17;0.14:0.10;0.19;0.0;0.10;0.03;0.07;0.23;0.0;0.0;0.07
1:0.16;0.13;0.43;0.99;0.04:0.0;0.0;0.0;0.02;0.0;0.0;0.02;0.0;0.14;0.32
2:0.48;0.56;0.04;0.96;0.18:0.10;0.13;0.0;0.09;0.07;0.0;0.0;0.0;0.01;0.19
3:0.62;0.19;0.13;0.43;0.29:0.0;0.15;0.0;0.12;0.11;0.0;0.01;0.03;0.0;0.08
...
4999:0.75;0.49;0.11;0.33;0.87:0.0;0.11;0.0;0.36;0.03;0.0;0.0;0.12;0.0;0.14
````

The format of each line is `N:page description:ads click rate`.

In this example you have a website of 5000 pages/articles, indexed by a number from 0 to 4999.

Each article is described by 5 variables, which could be the relation of this article with particular topics. For instance the first article, number 0, is 74% about politics, 83% about environment, 7% about technology, and so on, while the second article, number 1, is only 16% about politics, 13% about environment, and 43% about technology. 

Following the page description we find the ads click rate. Here we have 10 different ads. According to the previous excerpt, 10% of users that visited the article number 0 clicked on the ad number 0, and most clicked ad on this article is ad number 6 with a 23% click rate. 
On the other hand readers of article number 1 preferred ad number 9: 32% of these readers clicked on this ad.

## Our goal

We need to 





