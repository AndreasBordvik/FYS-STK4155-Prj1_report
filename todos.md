# Project 1 todos


## Exercise 1

## Exercise 2
feil konvergerer som 1/sqrt(n) (kanskje). (MonteCarlo)
sett i lys av computational heavy

What separates Machine Learning from optimization
- low generalization error as well (Goodfellow p.107).
- Motivation for bootstrap (and kfold CV)

note about iid assumption

underfitting and overfitting

## Exercise 3

Complexity degree 6, low variance between kfold splits

Compare bootstrap MSE for one degree with all kfold mean MSE for one degree.

Støyete vs. lite støyete datasett, ("Det har ikke så mye å si")
Usikkerheten går opp med flere og flere folds, for antall datapunkter går ned. 

Test bootstrap for n > 100. MSE for bootstrap blir mye bedre fordi sjansen for å sample samme punkt 2 
ganger blir mye mindre med større dataset! 


## Exercise 4-5
Relate model selection to the no free lunch theorem
(scikit p.33, Goodfellow p.113)

Section 5.2.2 in Goodfellow

vår fit opp mot sklearn
bootstrap for varierende degree lambda
CV for degree lambda
studere effekt med og uten scaling
studere effekt av scaling med og uten intercept

plotte vertikal linje for optimal lambda i beta-plot

Note that $\lambda$ is regularization parameter
Discuss the importance of regularization, for feature selection. 
Discuss bootstrap and cv with and without regularization.

Note regularization both with and without noise.

## Exercise 6
Add noise to z for all subtasks

Tenk på funksjonen man fitter som en kompresjon av terrenget. 

Curse of dimensionality, "The more dimensions the training data set has, the greater the risk of overfitting is"[Hands on scikit p215]