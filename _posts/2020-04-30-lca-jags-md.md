---
layout: post
published: false
title: lca-jags.md
---
# Tutorial for Latent Class Analysis with JAGS

In this tutorial, we use JAGS through the R package `rjags`. The example dataset can be found in the `poLCA` package in R, and it is simply called `cheating`.


## The Cheating Dataset

Let's start with the simplest model: just 2 latent classes. 

\[I am calling it *simple* but of course it is all relative. It took me while to understand how latent classes work and how they relate to my observed data. So compared to 3 or more classes, yes, the 2-class model is "simple".\]

**So let's get to it.**

Suppose you want to learn about cheating behavior in 319 undergraduates, so you assign them a self-report survey. You tell them the survey is *completely anonymous*, so they don't have to worry and should answer as honestly as possible. Here are the questions you asked:

| Variable Name | Have you... | 
|-------------|------------|    
| LIEEXAM |lied to avoid taking an exam? | 
| LIEPAPER |lied to avoid handing a term paper in on time? |
| FRAUD | purchased a term paper to hand in as their own or had obtained a copy of an exam prior to taking the exam? |
| COPYEXAM |copied answers during an exam from someone sitting near to them? | 

Students responded either (1) no or (2) yes.

You suspect some students may be more prone to cheating than others. But also that there are different *kinds of cheating*. 

Some people may copy answers from someone, while others may actively *hack the system to change their grades*. From the responses to the survey, you wish to understand patterns of cheating.

*You don't really care about those fine-grained of scores*; you are mostly interested in **summarizing** the main behaviors from the data. This is a case when latent class analysis may be appropriate.
You get back the scored exams and have item-level information about correct/incorrect, scored as 1 or 0, respectively.


## Latent Class Analysis with Two Classes
Here are some equations for such model:

![equation](http://tex.s2cms.ru/svg/%5Cbegin%7Balign*%7D%20%5Clabel%7Blca-likelihood%7D%0A%20f(y)%09%26%20%3Df(y_1%2Cy_2%2C...%2Cy_J%20%7C%20z%20%3D%201)%20%5Cboldsymbol%7BP(z%3D1)%7D%20%2B%20f(y_1%2Cy_2%2C...%2Cy_J%20%7C%20z%20%3D%202)%20%5Cboldsymbol%7BP(z%3D2)%7D%20%5Cnonumber%20%5C%5C%0A%20%09%09%20%20%20%20%20%20%09%20%20%20%20%20%09%20%20%20%20%20%09%26%20%3D%20f(y_1%7Cz%3D1)f(y_2%7C%20z%20%3D%201)...f(y_J%7Cz%3D1)%20%5Cboldsymbol%7BP(z%3D1)%7D%20%2B%20f(y_1%7Cz%3D2)f(y_2%7C%20z%20%3D%202)...f(y_J%7Cz%3D2)%20%5Cboldsymbol%7BP(z%3D2)%7D%5Cnonumber%5C%5C%0A%09%09%09%09%09%26%20%3D%20%5Cleft%5B%20%5CPi_%7Bj%3D1%7D%5EJ%20f(y_j%20%7C%20z%3D1)%20%5Cright%5D%20%5Cboldsymbol%7B%5Cpi_1%7D%20%2B%20%20%5Cleft%5B%20%5CPi_%7Bj%3D1%7D%5EJ%20f(y_j%20%7C%20z%3D2)%20%5Cright%5D%5Cboldsymbol%7B%5Cpi_2%7D%20%5Cnonumber%20%5C%5C%0A%09%09%09%09%09%26%20%3D%20%5Cleft%5B%20%5CPi_%7Bj%3D1%7D%5EJ%20p_%7Bj1%7D%5E%7By_%7Bj%7D%7D%20(1-p_%7Bj1%7D)%5E%7B1-y_%7Bj%7D%7D%20%5Cright%5D%20%5Cboldsymbol%7B%5Cpi_1%7D%20%2B%20%5Cleft%5B%20%5CPi_%7Bj%3D1%7D%5EJ%20p_%7Bj2%7D%5E%7By_%7Bj%7D%7D%20(1-p_%7Bj2%7D)%5E%7B1-y_%7Bj%7D%7D%20%5Cright%5D%20%5Cboldsymbol%7B%5Cpi_2%7D%5Cend%7Balign*%7D)

First specify the model in JAGS syntax:

```
model{
        for (i in 1:N){
            for (j in 1:J){
                y[i,j] ~ dbern(p[i,j]) # probability conditional on class
                p[i,j] <- z[i] * ilogit(theta[i]-b[j]) + (1-z[i]) * ilogit(theta[i]-b[j]+d[j])
            }
            theta[i] ~ dnorm(0,1) # prior on theta sets the scale
            alpha[i] <- ilogit(beta0 + beta1 * x[i])
            z[i] ~ dbern(alpha[i]) # class membership is 1 with prob. alpha
        }
        for (j in 1:J){
            d[j] ~ dnorm(mud,sigmad) # non-informative prior for d
            #    b[j,2] ~ dunif(-4,4)
        }
        mud ~ dnorm(0,1)
        isigmad ~ dgamma(100,100)
        sigmad <- 1/isigmad
        # alpha ~ dbeta(1,1) # prior for alpha follows Beta or dirich. distr.
        beta0 ~ dnorm(mu0,sigma0)
        mu0 ~ dnorm(0,1)
        isigma0 ~ dgamma(100,100)
        sigma0 <- 1/isigma0
        beta1 ~ dnorm(mu1,sigma1)
        mu1 ~ dnorm(0,1)
        isigma1 ~ dgamma(100,100)
        sigma1 <- 1/isigma1
    }
```

```R
mod2.string <- '
    model{
        for (i in 1:N){
            for (j in 1:J){
                y[i,j] ~ dbern(p[i,j]) # probability conditional on class
                p[i,j] <- z[i] * ilogit(theta[i]-b[j]) + (1-z[i]) * ilogit(theta[i]-b[j]+d[j])
            }
            theta[i] ~ dnorm(0,1) # prior on theta sets the scale
            alpha[i] <- ilogit(beta0 + beta1 * x[i])
            z[i] ~ dbern(alpha[i]) # class membership is 1 with prob. alpha
        }
        for (j in 1:J){
            d[j] ~ dnorm(mud,sigmad) # non-informative prior for d
            #    b[j,2] ~ dunif(-4,4)
        }
        mud ~ dnorm(0,1)
        isigmad ~ dgamma(100,100)
        sigmad <- 1/isigmad
        # alpha ~ dbeta(1,1) # prior for alpha follows Beta or dirich. distr.
        beta0 ~ dnorm(mu0,sigma0)
        mu0 ~ dnorm(0,1)
        isigma0 ~ dgamma(100,100)
        sigma0 <- 1/isigma0
        beta1 ~ dnorm(mu1,sigma1)
        mu1 ~ dnorm(0,1)
        isigma1 ~ dgamma(100,100)
        sigma1 <- 1/isigma1
    }
    '
    mod2 <- textConnection(mod2.string)
    jags.dat <- list("N"=N,"J"=J,"y"=y,"b"=b[,1], "x"=x)
    ## specify initial values
    jags.inits <- list()
    jags.inits[[1]] <- list(d=rep(0, J), beta0=0, beta1=.5,
                            mu0=0, mu1=0, mud=0, isigma0=1, isigma1=1, isigmad=1)

    mod.obj <- rjags::jags.model(file = mod2,
                                 data = jags.dat,
                                 n.chains = 1,
                                 n.adapt = burnin,
                                 inits=jags.inits)
    ##iter=5000
    fit <- rjags::coda.samples(model = mod.obj,
                               variable.names = c("d","theta","beta0","beta1",
                                                  "mu0","mu1","sigma0","sigma1","mud", "sigmad", "alpha"),
                               n.iter = iter)
```

## Latent Variable with Three Classes

## References
Martyn Plummer (2018). `rjags`: Bayesian Graphical Models using MCMC. R package version 4-8. https://CRAN.R-project.org/package=rjags