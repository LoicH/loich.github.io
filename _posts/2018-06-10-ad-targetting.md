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

We are managing a news website with lots of articles and pages, visited by many users. We want to monetize our content by placing ads on our website, but we can't randomly assign ads to articles, we need to find a way to put relevant ads in articles. 
Let's say everytime a user requests a page we can choose which ad the user will see, and we can record whether he clicked on the ad.

How can we maximize the number of clicks on our ads?

## Multi-armed bandits

This is a multi-armed bandit problem: at each turn we can choose one action (one ad) from a limited choice, without knowledge of the reward associated to each action. We can only learn from past experiences what actions are the most lucrative, to choose the future actions.

### What is the multi-armed bandit problem?

Let's assume we are in a weird casino with some free bandit machines in front of us. These machines do not require us to put money inside, we juste have to pull the arm and we could receive money. We have no knowledge about these machines, there may be one machine that will give us 1$ every time we press a button, and another machine that give us 100$ once every 10 pushes... 
The question is the following: "Without knowledge of the outcomes and their probabilities, what strategy is the best to maximize my gains?"

We could use 5 times each machine, then keep using the one that is the most lucrative. In our case, we use the 1st machine 5 times, we get 1$ everytime, then we use the 2nd machine 5 times, and there is a good chance we get nothing out of this machine. So now we think that the best machine is the 1st one, and we only use it. 

This is not optimal, we need to find a compromise between **exploiting** the best machine we found so far, and **exploring** to find a better machine.

Fortunately there are several efficient algorithms to maximize our gains.

#### Exploration, then exploitation ("Epsilon-first strategy")

I explained this method in the previous example, this method consists in first trying the various actions, then always selecting the best action. When doing this method we hope that we accurately modelled each machine's behaviour.

#### Exploitation, and a bit of exploration ("Epsilon-greedy strategy")

We select `e` between 0 and 1, 0.1 for instance, and with a probability of `e` we select a random machine, and the rest of the time (with a probability of `1-e`) we select the best machine we found so far. This is a good compromise between exploiting the best machine, and exploring. But this method depends on the choice of `e`: how can we find the best `e`?

#### An efficient mix of exploration and exploitation: UCB

(UCB stands for *Upper Confidence Bound*, from the paper [Finite-time Analysis of the Multiarmed Bandit Problem](https://rd.springer.com/article/10.1023%2FA%3A1013689704352)

The algorithm is pretty simple, at every turn we need to play the machine that maximizes $$x^2$$



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
