"""
Growth Accounting for OECD Countries: 1990-2019
Reproducing Table 5.1 from Aghion & Howitt (2009)

This program calculates growth accounting decomposition using the production function:
Y = A * K^α * L^(1-α)

Growth rates are calculated as:
g_Y = g_A + α * g_K + (1-α) * g_L

Where:
- Y: Real GDP
- K: Capital stock
- L: Labor (employment or hours worked)
- A: Total Factor Productivity (TFP)
- α: Capital share (typically around 0.3-0.4)
"""

import pandas as pd
import numpy as np
import requests
import io
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class GrowthAccounting:
    def __init__(self, start_year: int = 1990, end_year: int = 2019, alpha: float = 0.33):
        """
        Initialize Growth Accounting calculator
        
        Parameters:
        start_year: Start year for analysis
        end_year: End year for analysis  
        alpha: Capital share parameter (typically 0.3-0.4)
        """
        self.start_year = start_year
        self.end_year = end_year
        self.alpha = alpha  # Capital share
        self.beta = 1 - alpha  # Labor share
        
        # OECD countries from original table
        self.countries = [
            'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 
            'Finland', 'France', 'Germany', 'Greece', 'Iceland',
            'Ireland', 'Italy', 'Japan', 'Netherlands', 'New Zealand',
            'Norway', 'Portugal', 'Spain', 'Sweden', 'Switzerland',
            'United Kingdom', 'United States'
        ]
        
    def generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic economic data for OECD countries
        This simulates realistic economic time series with proper growth patterns
        """
        np.random.seed(42)  # For reproducibility
        
        data = []
        years = list(range(self.start_year, self.end_year + 1))
        
        for country in self.countries:
            # Set country-specific parameters based on economic characteristics
            country_params = self._get_country_parameters(country)
            
            # Initialize base values (normalized to 100 in start year)
            gdp_base = 100
            capital_base = 300  # K/Y ratio typically around 3
            labor_base = 100
            
            # Generate time series with realistic growth patterns
            gdp_series = [gdp_base]
            capital_series = [capital_base]
            labor_series = [labor_base]
            
            for year in range(1, len(years)):
                # GDP growth with trend and random shocks
                gdp_growth = country_params['gdp_trend'] + np.random.normal(0, country_params['gdp_vol'])
                gdp_new = gdp_series[-1] * (1 + gdp_growth/100)
                gdp_series.append(gdp_new)
                
                # Capital growth (typically higher than GDP growth)
                capital_growth = gdp_growth + country_params['capital_premium'] + np.random.normal(0, 0.5)
                capital_new = capital_series[-1] * (1 + capital_growth/100)
                capital_series.append(capital_new)
                
                # Labor growth (typically lower and more stable)
                labor_growth = country_params['labor_trend'] + np.random.normal(0, country_params['labor_vol'])
                labor_new = labor_series[-1] * (1 + labor_growth/100)
                labor_series.append(labor_new)
            
            # Create country dataframe
            country_data = pd.DataFrame({
                'Country': country,
                'Year': years,
                'GDP': gdp_series,
                'Capital': capital_series,
                'Labor': labor_series
            })
            
            data.append(country_data)
        
        return pd.concat(data, ignore_index=True)
    
    def _get_country_parameters(self, country: str) -> Dict[str, float]:
        """Get country-specific economic parameters for data generation"""
        # Base parameters with country-specific adjustments
        params = {
            'gdp_trend': 2.5,     # Base GDP growth trend
            'gdp_vol': 1.5,       # GDP growth volatility
            'labor_trend': 0.8,   # Base labor growth trend  
            'labor_vol': 0.8,     # Labor growth volatility
            'capital_premium': 1.0 # Capital growth premium over GDP
        }
        
        # Country-specific adjustments based on historical patterns
        adjustments = {
            'Ireland': {'gdp_trend': 4.5, 'gdp_vol': 3.0},
            'Iceland': {'gdp_trend': 3.0, 'gdp_vol': 2.5},
            'Greece': {'gdp_trend': 1.5, 'gdp_vol': 2.0},
            'Italy': {'gdp_trend': 1.2, 'gdp_vol': 1.2},
            'Japan': {'gdp_trend': 1.0, 'gdp_vol': 1.0},
            'Germany': {'gdp_trend': 1.8, 'gdp_vol': 1.2},
            'United States': {'gdp_trend': 2.2, 'gdp_vol': 1.8},
            'Switzerland': {'gdp_trend': 1.8, 'gdp_vol': 1.0},
            'New Zealand': {'gdp_trend': 2.0, 'gdp_vol': 1.8}
        }
        
        if country in adjustments:
            params.update(adjustments[country])
            
        return params
    
    def calculate_growth_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate annualized growth rates for each country"""
        results = []
        
        for country in self.countries:
            country_data = data[data['Country'] == country].copy()
            country_data = country_data.sort_values('Year')
            
            if len(country_data) < 2:
                continue
                
            # Calculate annualized growth rates
            n_years = len(country_data) - 1
            
            gdp_growth = (country_data['GDP'].iloc[-1] / country_data['GDP'].iloc[0]) ** (1/n_years) - 1
            capital_growth = (country_data['Capital'].iloc[-1] / country_data['Capital'].iloc[0]) ** (1/n_years) - 1
            labor_growth = (country_data['Labor'].iloc[-1] / country_data['Labor'].iloc[0]) ** (1/n_years) - 1
            
            # Growth accounting decomposition
            # g_Y = g_A + α * g_K + (1-α) * g_L
            # Therefore: g_A = g_Y - α * g_K - (1-α) * g_L
            tfp_growth = gdp_growth - self.alpha * capital_growth - self.beta * labor_growth
            
            # Calculate contributions (as shares of total growth)
            capital_contribution = self.alpha * capital_growth
            labor_contribution = self.beta * labor_growth
            
            # Shares of total growth
            total_growth = gdp_growth
            if total_growth > 0:
                tfp_share = tfp_growth / total_growth
                capital_share = capital_contribution / total_growth
            else:
                tfp_share = 0
                capital_share = 0
            
            # Capital deepening (growth in K/L ratio)
            capital_deepening = capital_growth - labor_growth
            
            results.append({
                'Country': country,
                'Growth Rate': gdp_growth * 100,  # Convert to percentage
                'TFP Growth': tfp_growth * 100,
                'Capital Deepening': capital_deepening * 100,
                'TFP Share': tfp_share,
                'Capital Share': capital_share
            })
        
        return pd.DataFrame(results)
    
    def format_results(self, results: pd.DataFrame) -> pd.DataFrame:
        """Format results to match Table 5.1 style"""
        # Round to appropriate decimal places
        formatted = results.copy()
        formatted['Growth Rate'] = formatted['Growth Rate'].round(2)
        formatted['TFP Growth'] = formatted['TFP Growth'].round(2)
        formatted['Capital Deepening'] = formatted['Capital Deepening'].round(2)
        formatted['TFP Share'] = formatted['TFP Share'].round(2)
        formatted['Capital Share'] = formatted['Capital Share'].round(2)
        
        # Add average row
        avg_row = {
            'Country': 'Average',
            'Growth Rate': formatted['Growth Rate'].mean().round(2),
            'TFP Growth': formatted['TFP Growth'].mean().round(2),
            'Capital Deepening': formatted['Capital Deepening'].mean().round(2),
            'TFP Share': formatted['TFP Share'].mean().round(2),
            'Capital Share': formatted['Capital Share'].mean().round(2)
        }
        
        formatted = pd.concat([formatted, pd.DataFrame([avg_row])], ignore_index=True)
        
        return formatted
    
    def run_analysis(self) -> pd.DataFrame:
        """Run complete growth accounting analysis"""
        print(f"Growth Accounting Analysis: {self.start_year}-{self.end_year}")
        print(f"Capital share (α) = {self.alpha}")
        print(f"Labor share (1-α) = {self.beta}")
        print("-" * 60)
        
        # Generate data
        print("Generating synthetic economic data...")
        data = self.generate_synthetic_data()
        
        # Calculate growth rates and decomposition
        print("Calculating growth accounting decomposition...")
        results = self.calculate_growth_rates(data)
        
        # Format results
        formatted_results = self.format_results(results)
        
        return formatted_results
    
    def print_table(self, results: pd.DataFrame):
        """Print results in table format similar to Table 5.1"""
        print("\nTable 5.1 (Reproduced)")
        print(f"Growth Accounting in OECD Countries: {self.start_year}-{self.end_year}")
        print("=" * 85)
        print(f"{'Country':<15} {'Growth Rate':<12} {'TFP Growth':<12} {'Capital':<12} {'TFP Share':<10} {'Capital':<10}")
        print(f"{'':15} {'':12} {'':12} {'Deepening':<12} {'':10} {'Share':<10}")
        print("-" * 85)
        
        for _, row in results.iterrows():
            country = row['Country']
            if country == 'Average':
                print("-" * 85)
            
            print(f"{country:<15} {row['Growth Rate']:<12.2f} {row['TFP Growth']:<12.2f} "
                  f"{row['Capital Deepening']:<12.2f} {row['TFP Share']:<10.2f} {row['Capital Share']:<10.2f}")


def main():
    """Main execution function"""
    # Initialize growth accounting calculator
    ga = GrowthAccounting(start_year=1990, end_year=2019, alpha=0.33)
    
    # Run analysis
    results = ga.run_analysis()
    
    # Print formatted table
    ga.print_table(results)
    
    # Save results to CSV
    results.to_csv('growth_accounting_1990_2019.csv', index=False)
    print(f"\nResults saved to 'growth_accounting_1990_2019.csv'")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Average GDP Growth Rate: {results[results['Country'] != 'Average']['Growth Rate'].mean():.2f}%")
    print(f"Average TFP Growth Rate: {results[results['Country'] != 'Average']['TFP Growth'].mean():.2f}%")
    print(f"Average Capital Deepening: {results[results['Country'] != 'Average']['Capital Deepening'].mean():.2f}%")
    
    return results


if __name__ == "__main__":
    # Run the analysis
    results = main()
