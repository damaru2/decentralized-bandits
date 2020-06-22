import numpy as np
import pickle
from utils import *
import os


def DDUCB(P, n, T, lambda_2, distributions, epsilon, eta, variance, mu_1, accel=False, C_small=False):
    regret = [0]
    lambda_2 = np.abs(lambda_2)
    K = len(distributions)
    C = None
    if accel:
        if C_small:
            C = int(np.ceil(np.log(2*n/epsilon)/np.sqrt(2*np.log(1/lambda_2))))
            C /= 3
            C = int(C)
        else:
            C = int(np.ceil(np.log(2*n/epsilon)/np.sqrt(2*np.log(1/lambda_2))))
    else:
        if C_small:
            C = int(np.ceil(np.log(2*n/epsilon)/np.sqrt(2*np.log(1/lambda_2))))
        else:
            C = int(np.ceil(np.log(n/epsilon)/np.log(1/lambda_2)))
    print('C: {}'.format(C))

    first_pulls = np.array([np.array([distributions[i]() for i in range(K)]) for j in range(n)])
    beta = RewNumPulls(n, rewards_array=first_pulls, pulls_array=np.ones((n, K)))
    alpha = RewNumPulls(n,rewards_array=beta.rewards/K, pulls_array=beta.pulls/K)
    gamma = RewNumPulls(n, rewards_array=np.zeros((n, K)), pulls_array=np.zeros((n, K)))
    delta = RewNumPulls(n, rewards_array=np.zeros((n, K)), pulls_array=np.zeros((n, K)))

    unaccel_mixer = None
    if accel:
        # Sparse should be better here
        mixer = AccelMix(P, lambda_2, C, sparse=True)
    else:
        # It is not clear if sparsity is better in this case
        mixer = UnaccelMix(P, lambda_2, C, sparse=False)
        #mixer = UnaccelMix(P, lambda_2, C, sparse=True)
    t = K
    s = K
    accum = None
    while t <= T:
        for r in range(C):
            accum = 0
            for i in range(n):
                UCBs = np.array([alpha.rewards[i][k]/alpha.pulls[i][k] +
                                 np.sqrt(2*eta*variance*np.log(s)/(n*alpha.pulls[i][k])) for k in range(K)])
                k_star = np.argmax(UCBs)
                u = distributions[k_star]()
                accum += u
                gamma.rewards[i][k_star] += u
                gamma.pulls[i][k_star] += 1
                # It also works adding this
                #'''
                alpha.rewards[i][k_star] += u/n
                alpha.pulls[i][k_star] += 1/n
                s += 1
                #'''
            regret.append(regret[-1] + n*mu_1-accum)
            t += 1
            if t > T:
                print('Number of pulls per arm (best means first)')
                print(np.sum(alpha.pulls, axis=0))
                return regret
            beta.rewards = mixer.mix(beta.rewards)
            beta.pulls = mixer.mix(beta.pulls)

        # It also works adding this
        '''
        if not unaccel_mixer:
            unaccel_mixer = UnaccelMix(P, lambda_2, C, sparse=False)

        delta.rewards = unaccel_mixer.mix(delta.rewards)
        delta.pulls = unaccel_mixer.mix(delta.pulls)
        s += 1
        #'''
        s = (t-C)*n
        if accel:
            delta.rewards += beta.rewards
            delta.pulls += beta.pulls
            alpha.rewards = delta.rewards
            alpha.pulls = delta.pulls
            beta.rewards = gamma.rewards
            beta.pulls = gamma.pulls
            gamma.rewards = np.zeros((n, K))
            gamma.pulls = np.zeros((n, K))
        else:
            alpha.rewards = beta.rewards
            alpha.pulls = beta.pulls
            beta.rewards += gamma.rewards
            beta.pulls += gamma.pulls
            gamma.rewards = np.zeros((n, K))
            gamma.pulls = np.zeros((n, K))

    return regret


def landgren_bandits(P, n, T, lambda_2, distributions, epsilon_c, gamma, variance, mu_1):
    regret = [0]
    lambda_2 = np.abs(lambda_2)
    K = len(distributions)
    first_pulls = np.array([np.array([distributions[k]() for k in range(K)]) for j in range(n)])
    estim = RewNumPulls(n,rewards_array=first_pulls, pulls_array=np.ones((n,K)))
    t = 1
    accum = None
    UCBs = None
    mixer = UnaccelMix(P=P, lambda_2=lambda_2, C=1, sparse=True)
    while t <= T:
        accum = 0
        for i in range(n):
            k = 1
            UCBs = np.array([estim.rewards[i][k]/estim.pulls[i][k] +
                             np.sqrt((estim.pulls[i][k]+epsilon_c[i])*2*gamma*variance*np.log(t)/(n*(estim.pulls[i][k]**2))) for k in range(K)])
            k_star = np.argmax(UCBs)
            u = distributions[k_star]()
            accum += u
            estim.rewards[i][k_star] += u
            estim.pulls[i][k_star] += 1
            estim.rewards = mixer.mix(estim.rewards)
            estim.pulls = mixer.mix(estim.pulls)

        regret.append(regret[-1] + n*mu_1 -accum)
        t += 1

    print(np.sum(estim.pulls, axis=0))
    return regret


def main(n, T, save = True, rerun=False, type_P='cycle', dducb_in=[], landgren_in=[], mus=[1, 0.8]):
    print('n =',n, type_P)
    # The code assumes A defines a graph with regular degree and the matrix P is doubly stochastic,
    P = compute_P(n, type=type_P)

    print('computing constants')
    #'''
    filename = './data/constants_n{}_{}'.format(n, type_P)
    if os.path.isfile(filename):
        f = open(filename, "rb")
        epsilon_c = pickle.load(f)
        lambda_2 = pickle.load(f)
        f.close()
    else:
        epsilon_c, lambda_2 = compute_constants(P, n)
        fileObject = open(filename,'wb')
        pickle.dump(epsilon_c,fileObject)
        pickle.dump(lambda_2,fileObject)
        fileObject.close()
    #'''

    sigma = 1
    mus = sorted(mus, key=lambda x: -x)
    distributions = [lambda mu=mu, sigma=sigma:np.random.normal(mu, sigma) for mu in mus]

    #'''
    print('Running Landgren')
    gamma = 2
    landgren = []
    for gamma in landgren_in:
        filename = './data/landgren_n{}_t{}_{}_gamma{}_sigma{}'.format(n, T, type_P, gamma, sigma)
        if not rerun and os.path.isfile(filename):
             regret_landgren = read_regret(filename)
        else:
            regret_landgren = landgren_bandits(P, n, T, lambda_2, distributions, epsilon_c, gamma=gamma, variance=sigma*sigma, mu_1=mus[0])
            if save:
                fileObject = open(filename,'wb')
                pickle.dump(regret_landgren,fileObject)
                fileObject.close()
        landgren.append((regret_landgren, r'coopUCB $\gamma$={}'.format(gamma)))

    dducb = []
    #'''
    print('Running DDUCB')
    eta = 2
    epsilon= 1/22
    #epsilon= 1/7
    for C_small, accel in dducb_in:
        filename = './data/dducb_n{}_t{}_{}_{}{}_sigma{}'.format(n, T, type_P, 'accel' if accel else 'unaccel', '_Csmall' if C_small else '', sigma)
        if not rerun and os.path.isfile('./{}'.format(filename)):
            regret_dducb = read_regret(filename)
        else:
            regret_dducb = DDUCB(P, n, T, lambda_2, distributions, epsilon=epsilon, eta=eta, variance=sigma*sigma, mu_1=mus[0], accel=accel, C_small=C_small)

        if save:
            fileObject = open(filename,'wb')
            pickle.dump(regret_dducb,fileObject)
            fileObject.close()
        dducb.append((regret_dducb, 'DDUCB\_{}{}'.format('accel' if accel else 'unaccel', '\_Csmall' if C_small else '')))
    #'''
    all = landgren + dducb
    plot(all, n, T, type_P, sigma, K=len(distributions))
    return all

def main_aux():
    '''
    This function generates data and plots of just one execution.
    The main paper uses an average of 10 of these executions
    '''
    C_acc = (False, True) # (C, accel)
    Csmall_unacc = (True, False) # (C_small, unaccel)
    dducb_instances = [Csmall_unacc, C_acc]
    landgren_instances = [2, 1.01, 1.0001]
    mus = [1.0] + [0.8] * 16

    main(100, 10000, save=True, rerun=False, type_P='cycle', dducb_in=dducb_instances, landgren_in=landgren_instances, mus=mus)
    main(200, 10000, save=True, rerun=False, type_P='cycle', dducb_in=dducb_instances, landgren_in=landgren_instances, mus=mus)
    main(100, 10000, save=True, rerun=False, type_P='grid', dducb_in=dducb_instances, landgren_in=landgren_instances, mus=mus)
    main(225, 10000, save=True, rerun=False, type_P='grid', dducb_in=dducb_instances, landgren_in=landgren_instances, mus=mus)

if __name__ == "__main__":
    main_aux()
