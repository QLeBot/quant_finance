import numpy as np
from fund_portfolio_data import saa

class PortfolioRebalancer:
    def __init__(self, base_saa=saa):
        self.base_saa = base_saa
        self.asset_classes = list(base_saa.keys())
        
        # Define asset class characteristics
        self.asset_characteristics = {
            "fixed_income": {
                "risk_level": "low",
                "correlation_with_equity": -0.2,
                "liquidity": "high"
            },
            "equity": {
                "risk_level": "high",
                "correlation_with_equity": 1.0,
                "liquidity": "high"
            },
            "real_estate": {
                "risk_level": "medium",
                "correlation_with_equity": 0.4,
                "liquidity": "low"
            },
            "infrastructure": {
                "risk_level": "medium",
                "correlation_with_equity": 0.3,
                "liquidity": "low"
            },
            "timber_farmland": {
                "risk_level": "medium",
                "correlation_with_equity": 0.2,
                "liquidity": "low"
            },
            "private_equity": {
                "risk_level": "high",
                "correlation_with_equity": 0.7,
                "liquidity": "low"
            },
            "hedge_funds": {
                "risk_level": "medium",
                "correlation_with_equity": 0.5,
                "liquidity": "medium"
            },
            "metals_commodities": {
                "risk_level": "high",
                "correlation_with_equity": 0.3,
                "liquidity": "medium"
            },
            "cash": {
                "risk_level": "low",
                "correlation_with_equity": 0.0,
                "liquidity": "high"
            }
        }
        
        # Define scenario adjustments
        self.scenario_adjustments = {
            "Optimistic": {
                "risk_appetite": 1.2,  # Increase risk
                "liquidity_preference": 0.9,  # Slightly decrease liquidity
                "correlation_sensitivity": 0.8  # Less sensitive to correlation
            },
            "Base Case": {
                "risk_appetite": 1.0,
                "liquidity_preference": 1.0,
                "correlation_sensitivity": 1.0
            },
            "Pessimistic": {
                "risk_appetite": 0.8,  # Decrease risk
                "liquidity_preference": 1.1,  # Increase liquidity
                "correlation_sensitivity": 1.2  # More sensitive to correlation
            }
        }

    def calculate_risk_score(self, asset, scenario):
        """Calculate risk score for an asset based on scenario"""
        char = self.asset_characteristics[asset]
        scenario_params = self.scenario_adjustments[scenario]
        
        # Base risk score
        risk_scores = {"low": 1, "medium": 2, "high": 3}
        base_score = risk_scores[char["risk_level"]]
        
        # Adjust based on scenario
        adjusted_score = base_score * scenario_params["risk_appetite"]
        
        # Adjust for correlation sensitivity
        correlation_impact = char["correlation_with_equity"] * scenario_params["correlation_sensitivity"]
        
        # Adjust for liquidity preference
        liquidity_scores = {"high": 1, "medium": 1.2, "low": 1.5}
        liquidity_impact = liquidity_scores[char["liquidity"]] * scenario_params["liquidity_preference"]
        
        return adjusted_score * (1 + correlation_impact) * liquidity_impact

    def rebalance_portfolio(self, scenario, max_deviation=0.05):
        """Rebalance portfolio based on scenario"""
        # Calculate risk scores for all assets
        risk_scores = {asset: self.calculate_risk_score(asset, scenario) 
                      for asset in self.asset_classes}
        
        # Normalize risk scores to get target weights
        total_score = sum(risk_scores.values())
        target_weights = {asset: score/total_score for asset, score in risk_scores.items()}
        
        # Calculate maximum allowed deviation from base SAA
        adjusted_weights = {}
        for asset in self.asset_classes:
            base_weight = self.base_saa[asset]
            target_weight = target_weights[asset]
            
            # Limit deviation from base SAA
            if target_weight > base_weight:
                max_weight = min(base_weight * (1 + max_deviation), target_weight)
            else:
                max_weight = max(base_weight * (1 - max_deviation), target_weight)
            
            adjusted_weights[asset] = max_weight
        
        # Normalize to ensure weights sum to 1
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {asset: weight/total_weight 
                            for asset, weight in adjusted_weights.items()}
        
        return normalized_weights

    def get_rebalancing_rationale(self, scenario, new_allocation):
        """Generate explanation for the rebalancing decisions"""
        rationale = []
        
        # Compare with base SAA
        for asset in self.asset_classes:
            change = new_allocation[asset] - self.base_saa[asset]
            if abs(change) > 0.001:  # Only report meaningful changes
                direction = "increased" if change > 0 else "decreased"
                rationale.append(
                    f"{asset.replace('_', ' ').title()}: {direction} by "
                    f"{abs(change)*100:.1f}% due to {scenario} scenario"
                )
        
        return rationale 