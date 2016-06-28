from util import *

class CoinFlipScenario:
        
    def __init__(self, n_pub, n_pri, players):
        self.n_p = n_pub
        self.n_v = n_pri
        self.k = players
        
    # Set the number of public heads observation in this scenario
    def set_public_obs(self, n_heads):
        self.h_p = n_heads
        
    def individual_posterior_params(self, h_v):
        a = 1 + self.h_p + h_v
        b = 1 + (self.n_p - self.h_p) + (self.n_v - h_v)
        return (a, b)
        
    # Get the individual posterior after seeing private information
    def individual_posterior_private(self, h_v):
        (a, b) = self.individual_posterior_params(h_v)
        return scipy.stats.beta(a, b)
    
    # Beta-binomial distribution over private heads given public observation
    def posterior_prob_public(self):
        a = 1 + self.h_p
        b = 1 + self.n_p - self.h_p
        
        return beta_binomial_probs(self.n_v, a, b)
        
    # Beta-binomial distribution over someone else's heads given public and private observations
    def posterior_prob_signal(self, h_v):
        a = 1 + self.h_p + h_v
        b = 1 + (self.n_p - self.h_p) + (self.n_v - h_v)
        
        return beta_binomial_probs(self.n_v, a, b)

    def get_interval_left(self, myReport, x, strat_other):
        left_end = np.maximum(0, 2 * x - myReport)
        interval_left = np.where(x <= myReport, left_end, myReport).reshape(-1,1)
        return interval_left
    
    def get_interval_right(self, myReport, x, strat_other):
        right_end = np.minimum(1, 2 * x - myReport)        
        interval_right = np.where(x <= myReport, myReport, right_end).reshape(-1,1)
        return interval_right

    # Binary matrix of whether or not some report is outside the interval
    def get_outside_interval(self, x, strat_other, interval_left, interval_right):
        report_matrix = np.tile(strat_other, (len(x), 1))
        
        outside_interval = np.logical_or(
            np.logical_and(np.less(report_matrix, interval_left), np.logical_not(np.isclose(report_matrix, interval_left))),
            np.logical_and(np.greater(report_matrix, interval_right), np.logical_not(np.isclose(report_matrix, interval_right))),
        )
        return outside_interval
    
    # Binary matrix of whether or not some report is on the interval boundary
    def get_on_interval_boundary(self, x, strat_other, interval_left, interval_right):
        report_matrix = np.tile(strat_other, (len(x), 1))
        
        on_interval_boundary = np.logical_or(
            np.isclose(report_matrix, interval_left),
            np.isclose(report_matrix, interval_right)
        )
        return on_interval_boundary
    
    def prob_other_signal(self, x, h_v):

        prob_mat = np.ones((len(x), self.n_v+1))
        for h in range(self.n_v+1):
            prob_array = (x ** h) * ((1-x) ** (self.n_v - h))
            prob_mat[:, h] = prob_array
        return prob_mat
    
    # Get probability of winning given true bias x, my strategy, others' strategy
    # NOTE: this is my belief about winning, but may not actually reflect my true probability of winning
    def my_prob_win(self, x, h_v, my_report, strat_other):

        my_posterior_probs = self.prob_other_signal(x, h_v)
        
        
        # compute interval, I lose if other reports are strictly within this interval
        interval_left = self.get_interval_left(my_report, x, strat_other)
        interval_right = self.get_interval_right(my_report, x, strat_other)

        # Binary matrix of whether or not another report is on interval boundary
        outside_interval = self.get_outside_interval(x, strat_other, interval_left, interval_right)       
        prob_outside_interval = np.sum(my_posterior_probs * outside_interval, axis=1)
            
        # Binary matrix of whether or not another report is on interval boundary
        on_interval_boundary = self.get_on_interval_boundary(x, strat_other, interval_left, interval_right)
        prob_on_boundary = np.sum(my_posterior_probs * on_interval_boundary, axis=1)
        
        totalProb = 0
        for i in range(self.k):
            # We're looking at (k-1 choose i) here
            totalProb += scipy.misc.comb(self.k-1, i) * \
            (prob_outside_interval ** (self.k-1-i)) * (prob_on_boundary ** i) / (i+1)

        return totalProb        
     
    # Expected payoff given a private signal
    def expected_payoff(self, h_v, my_report, strat_other):

        # function to integrate on
        posterior = self.individual_posterior_private(h_v)
        func = lambda x: self.my_prob_win(x, h_v, my_report, strat_other) * posterior.pdf(x)
        
        payoff, err = integrate.quadrature(func, 0, 1,\
                            tol=1.49e-10, rtol=1.49e-10, maxiter=600, miniter=200, vec_func=True)
        return payoff                    
    
    # Expected payoff ex ante (should be 1/k when strategies are symmetric)
    def ex_ante_payoff(self, strat_mine, strat_other):
        signal_probs = self.posterior_prob_public()
        payoffs = np.ones(signal_probs.size)
        for i, _ in enumerate(signal_probs):
            payoffs[i] = self.expected_payoff(i, strat_mine[i], strat_other)
        return np.dot(signal_probs, payoffs), payoffs

    ########################
    # Helpful functions for testing
    ########################
    
    def truthful_strategy_mode(self):
        return (self.h_p + np.linspace(0, self.n_v, self.n_v + 1)) / (self.n_p + self.n_v)
    
    # Generate a strategy that weights the public and private modes
    def gen_weighted_strategy(self, w_pub):        
        vec_report_pub = self.h_p / self.n_p  
        vec_report_priv = np.linspace(0, 1, self.n_v+1)

        return w_pub * vec_report_pub + (1 - w_pub) * vec_report_priv

# Get x values where the prob_win function is discontinuous
def get_discontinuities(my_report, strat_other) :   
    discontinuities = [0] * (len(strat_other) - 1)
    index = 0
    for r in strat_other:
        if (np.isclose(r, my_report)):
            continue
        else:
            discontinuities[index] = (my_report + r) / 2
            index += 1
    return discontinuities

class CoinFlipScenario_Piecewise(CoinFlipScenario):
    
    def expected_payoff(self, h_v, my_report, strat_other):        
        # find all discontinuities
        discon = get_discontinuities(my_report, strat_other)
        discon.insert(0,0)
        discon.append(1)
        
        # function to integrate on
        posterior = self.individual_posterior_private(h_v)
        func = lambda x: self.my_prob_win(x, h_v, my_report, strat_other) * posterior.pdf(x)
        
        # integrate continuous pieces separately
        total_payoff = 0
        for i in range(len(discon)-1):
            payoff, err = integrate.quadrature(func, discon[i], discon[i+1],\
                            tol=1.49e-10, rtol=1.49e-10, maxiter=600, miniter=200, vec_func=True)
            total_payoff += payoff
        return total_payoff

# Fast version of integral using closed form calculations
class CoinFlipScenario_Betainc(CoinFlipScenario):
        
    def expected_payoff(self, h_v, my_report, strat_other):
        
        (a, b) = self.individual_posterior_params(h_v) # my         
            
        payoff = 0
        
        # print my_report, strat_other
        
        # Go thru all combinations of others' reports
        for other_signals in multichoose(self.k - 1, range(len(strat_other))):
            sigs = np.array(other_signals)
            # sum number of people with each report
            bins = np.bincount(sigs, minlength=self.n_v + 1)   
            
            h = sigs.sum() # number of heads observed in total by others: sum_i h_i
            t = (self.k - 1) * self.n_v - h # number of tails observed in total by others: sum_i n_v - h_i
            alpha = a + h 
            beta = b + t
            
            # number of permutations for this combination            
            c = scipy.misc.factorial(self.k - 1) / scipy.misc.factorial(bins).prod()
            
            # number of other reports I am tied with (exactly)            
            k = 1
            try:
                # note that bins[strat_other == my_report] doesn't work when my report is different from all bins
                k = np.sum(bins * (strat_other == my_report)) + 1
            except:                
                pass
                            
            # print strat_other, sigs
            # lower, upper bounds of interval over which I win
            reps = strat_other[sigs]     
            
            l = 0
            try: 
                l = ( np.amax(reps[reps < my_report]) + my_report ) / 2
            except:
                pass
                
            u = 1            
            try:            
                u = ( np.amin(reps[reps > my_report]) + my_report ) / 2                
            except:
                pass
            
            # diagnostics
            # print sigs, bins, c, k, l, u
            
            p = scipy.special.comb(self.n_v, sigs).prod()
            
            # Because the betainc function is regularized, we have to adjust it
            total = 1.0 * c / k * p * scipy.special.beta(alpha, beta) / scipy.special.beta(a, b) * \
                (scipy.special.betainc(alpha, beta, u) - scipy.special.betainc(alpha, beta, l))
                
            # This may result in numerical issues
            payoff = payoff + total
            
        return payoff
    
