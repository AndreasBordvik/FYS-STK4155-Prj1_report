# Project 1 breakdown

## Pre Exercise
"Our first step will be to perform an OLS regression analysis of this function, trying out a polynomial fit with an x and y dependence of the form [x,y,x2,y2,xy,…]." - Hva mener Morten med dette? Skal vi fitte interactions?
## Exercise 1
Write own code for OLS for polynomials up to fifth order
+ Using matrix inversion
+ Using SVD

Find the confidence intervals of the estimators, computing variance, MSE and R2

The code has to include scaling for the data (explain why)

Split data into train test - *PRESNT CRITICAL DISUSSION ON SCALING!*

## EXERCISE 2

+   Morten "we want you to show this equation:"[lecture 17.sept, 47:50 min]
    $$
    \mathbb{E}\left[(\boldsymbol{y}-\boldsymbol{\tilde{y}})^2\right]=\mathbb{E}\left[(\boldsymbol{y}-\mathbb{E}\left[\boldsymbol{\tilde{y}}\right])^2\right]+\mathrm{Var}\left[\boldsymbol{\tilde{y}}\right]+\sigma^2,
    $$

+   In bootstrap, ONLY touch train data, NOT test. only use test when you need to make 
    a prediction. 

+ Explain the bias-variance tradeoff
+ Give a critical discussion of the bias-variance tradeoff as function of our model complexity

