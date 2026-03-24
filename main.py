import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

class Option:
    def __init__(self, S, K, T, r, sigma, option_type):
        self.S = S  # Current stock price
        self.K = K  # Option strike price
        self.T = T  # Time to expiration
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility
        self.option_type = option_type  # 'call' or 'put'

    def black_scholes(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return price

    def greeks(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        delta = norm.cdf(d1) if self.option_type == 'call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
                 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d1 - self.sigma * np.sqrt(self.T)))
        rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d1)  # For call
        return delta, gamma, vega, theta, rho

    def monte_carlo(self, num_simulations):
        np.random.seed(42)
        payoffs = []
        for _ in range(num_simulations):
            ST = self.S * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma * np.sqrt(self.T) * np.random.normal())
            if self.option_type == 'call':
                payoffs.append(max(ST - self.K, 0))
            else:
                payoffs.append(max(self.K - ST, 0))
        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def implied_volatility(self, market_price, tol=1e-5, max_iter=100):
        sigma = 0.2  # initial guess
        for _ in range(max_iter):
            price = self.black_scholes()
            if abs(price - market_price) < tol:
                return sigma
            vega = self.greeks()[2]  # Vega
            sigma -= (price - market_price) / vega
        return sigma

    def plot_3d_surface(self):
        S = np.linspace(0.01, 2 * self.K, 100)
        sigma = np.linspace(0.01, 1, 100)
        X, Y = np.meshgrid(S, sigma)
        Z = np.array([[Option(s, self.K, self.T, self.r, y, self.option_type).black_scholes() for s in S] for y in sigma])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('Stock Price S')
        ax.set_ylabel('Volatility sigma')
        ax.set_zlabel('Option Price')
        plt.title('3D Option Pricing Surface')
        plt.show()

# Example usage:
# option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
# print(option.black_scholes())
# option.plot_3d_surface()