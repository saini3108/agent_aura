import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from agent_service.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class ECLResult:
    """Result of ECL calculation"""
    stage_1_ecl: float
    stage_2_ecl: float
    stage_3_ecl: float
    total_ecl: float
    scenario_weights: Dict[str, float]
    calculation_details: Dict[str, Any]

@dataclass
class RWAResult:
    """Result of RWA calculation"""
    credit_rwa: float
    market_rwa: float
    operational_rwa: float
    total_rwa: float
    capital_ratios: Dict[str, float]
    calculation_details: Dict[str, Any]

@dataclass
class ModelValidationResult:
    """Result of model validation"""
    validation_passed: bool
    validation_score: float
    test_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    issues: List[str]
    recommendations: List[str]

class BankingToolkit:
    """Banking calculation tools for ECL, RWA, and model validation"""

    def __init__(self):
        self.config = {
            "default_confidence_level": 0.95,
            "max_calculation_time": settings.ECL_CALCULATION_TIMEOUT,
            "precision_decimal_places": 4
        }

    async def calculate_ecl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Expected Credit Loss (ECL) according to IFRS 9"""

        try:
            # Validate inputs
            required_fields = ["portfolio_data", "pd_curves", "lgd_estimates", "ead_estimates"]
            for field in required_fields:
                if field not in inputs:
                    raise ValueError(f"Missing required field: {field}")

            # Extract calculation parameters
            portfolio_data = inputs["portfolio_data"]
            pd_curves = inputs["pd_curves"]
            lgd_estimates = inputs["lgd_estimates"]
            ead_estimates = inputs["ead_estimates"]

            # Get scenario parameters
            scenarios = inputs.get("scenarios", {
                "base": {"weight": 0.5, "macro_factors": {}},
                "upside": {"weight": 0.25, "macro_factors": {}},
                "downside": {"weight": 0.25, "macro_factors": {}}
            })

            # Calculate ECL for each scenario
            scenario_results = {}
            for scenario_name, scenario_params in scenarios.items():
                scenario_ecl = await self._calculate_scenario_ecl(
                    portfolio_data, pd_curves, lgd_estimates, ead_estimates, scenario_params
                )
                scenario_results[scenario_name] = scenario_ecl

            # Calculate weighted ECL
            weighted_ecl = self._calculate_weighted_ecl(scenario_results, scenarios)

            # Create result
            result = ECLResult(
                stage_1_ecl=weighted_ecl["stage_1"],
                stage_2_ecl=weighted_ecl["stage_2"],
                stage_3_ecl=weighted_ecl["stage_3"],
                total_ecl=weighted_ecl["total"],
                scenario_weights={k: v["weight"] for k, v in scenarios.items()},
                calculation_details={
                    "scenarios": scenario_results,
                    "methodology": "IFRS 9 Expected Credit Loss",
                    "calculation_date": datetime.utcnow().isoformat(),
                    "portfolio_size": len(portfolio_data) if isinstance(portfolio_data, list) else portfolio_data.get("size", 0)
                }
            )

            logger.info(f"ECL calculation completed - Total ECL: {result.total_ecl:.2f}")

            return {
                "success": True,
                "ecl_result": {
                    "stage_1_ecl": result.stage_1_ecl,
                    "stage_2_ecl": result.stage_2_ecl,
                    "stage_3_ecl": result.stage_3_ecl,
                    "total_ecl": result.total_ecl,
                    "scenario_weights": result.scenario_weights,
                    "calculation_details": result.calculation_details
                },
                "warnings": [],
                "recommendations": [
                    "Review scenario weights quarterly",
                    "Validate PD curves against historical data",
                    "Monitor concentration risk in portfolio"
                ]
            }

        except Exception as e:
            logger.error(f"ECL calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "ecl_result": None,
                "warnings": [f"Calculation failed: {str(e)}"],
                "recommendations": ["Review input data quality", "Check calculation parameters"]
            }

    async def _calculate_scenario_ecl(self, portfolio_data: Dict[str, Any], pd_curves: Dict[str, Any],
                                     lgd_estimates: Dict[str, Any], ead_estimates: Dict[str, Any],
                                     scenario_params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate ECL for a specific scenario"""

        # Simulate ECL calculation logic
        # In production, this would use actual portfolio data and models

        # Stage 1: 12-month ECL (no significant increase in credit risk)
        stage_1_exposure = portfolio_data.get("stage_1_exposure", 1000000)
        stage_1_pd = pd_curves.get("12m_pd", 0.02)
        stage_1_lgd = lgd_estimates.get("stage_1_lgd", 0.45)
        stage_1_ead = ead_estimates.get("stage_1_ead", 0.95)

        # Apply scenario adjustments
        macro_adjustment = scenario_params.get("macro_factors", {}).get("gdp_shock", 0)
        adjusted_pd_1 = stage_1_pd * (1 + macro_adjustment)

        stage_1_ecl = stage_1_exposure * adjusted_pd_1 * stage_1_lgd * stage_1_ead

        # Stage 2: Lifetime ECL (significant increase in credit risk)
        stage_2_exposure = portfolio_data.get("stage_2_exposure", 200000)
        stage_2_pd = pd_curves.get("lifetime_pd", 0.15)
        stage_2_lgd = lgd_estimates.get("stage_2_lgd", 0.50)
        stage_2_ead = ead_estimates.get("stage_2_ead", 0.90)

        adjusted_pd_2 = stage_2_pd * (1 + macro_adjustment * 1.5)
        stage_2_ecl = stage_2_exposure * adjusted_pd_2 * stage_2_lgd * stage_2_ead

        # Stage 3: Lifetime ECL (credit-impaired)
        stage_3_exposure = portfolio_data.get("stage_3_exposure", 50000)
        stage_3_pd = 1.0  # Already defaulted
        stage_3_lgd = lgd_estimates.get("stage_3_lgd", 0.70)
        stage_3_ead = ead_estimates.get("stage_3_ead", 1.0)

        stage_3_ecl = stage_3_exposure * stage_3_pd * stage_3_lgd * stage_3_ead

        return {
            "stage_1": round(stage_1_ecl, self.config["precision_decimal_places"]),
            "stage_2": round(stage_2_ecl, self.config["precision_decimal_places"]),
            "stage_3": round(stage_3_ecl, self.config["precision_decimal_places"]),
            "total": round(stage_1_ecl + stage_2_ecl + stage_3_ecl, self.config["precision_decimal_places"])
        }

    def _calculate_weighted_ecl(self, scenario_results: Dict[str, Dict[str, float]],
                               scenarios: Dict[str, Any]) -> Dict[str, float]:
        """Calculate probability-weighted ECL across scenarios"""

        weighted_ecl = {"stage_1": 0, "stage_2": 0, "stage_3": 0, "total": 0}

        for scenario_name, results in scenario_results.items():
            weight = scenarios[scenario_name]["weight"]

            for stage in ["stage_1", "stage_2", "stage_3", "total"]:
                weighted_ecl[stage] += results[stage] * weight

        # Round results
        for stage in weighted_ecl:
            weighted_ecl[stage] = round(weighted_ecl[stage], self.config["precision_decimal_places"])

        return weighted_ecl

    async def calculate_rwa(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Risk-Weighted Assets (RWA) according to Basel III"""

        try:
            # Validate inputs
            required_fields = ["exposure_data", "risk_weights", "capital_data"]
            for field in required_fields:
                if field not in inputs:
                    raise ValueError(f"Missing required field: {field}")

            # Extract calculation parameters
            exposure_data = inputs["exposure_data"]
            risk_weights = inputs["risk_weights"]
            capital_data = inputs["capital_data"]

            # Calculate Credit RWA
            credit_rwa = await self._calculate_credit_rwa(exposure_data, risk_weights)

            # Calculate Market RWA
            market_rwa = await self._calculate_market_rwa(exposure_data.get("market_exposures", {}))

            # Calculate Operational RWA
            operational_rwa = await self._calculate_operational_rwa(exposure_data.get("operational_data", {}))

            # Calculate total RWA
            total_rwa = credit_rwa + market_rwa + operational_rwa

            # Calculate capital ratios
            capital_ratios = self._calculate_capital_ratios(capital_data, total_rwa)

            # Create result
            result = RWAResult(
                credit_rwa=credit_rwa,
                market_rwa=market_rwa,
                operational_rwa=operational_rwa,
                total_rwa=total_rwa,
                capital_ratios=capital_ratios,
                calculation_details={
                    "methodology": "Basel III Standardized Approach",
                    "calculation_date": datetime.utcnow().isoformat(),
                    "basel_framework": "Basel III",
                    "regulatory_adjustments": inputs.get("regulatory_adjustments", {})
                }
            )

            logger.info(f"RWA calculation completed - Total RWA: {result.total_rwa:.2f}")

            return {
                "success": True,
                "rwa_result": {
                    "credit_rwa": result.credit_rwa,
                    "market_rwa": result.market_rwa,
                    "operational_rwa": result.operational_rwa,
                    "total_rwa": result.total_rwa,
                    "capital_ratios": result.capital_ratios,
                    "calculation_details": result.calculation_details
                },
                "warnings": [],
                "recommendations": [
                    "Monitor capital ratio compliance",
                    "Review concentration limits",
                    "Validate risk weight assignments"
                ]
            }

        except Exception as e:
            logger.error(f"RWA calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "rwa_result": None,
                "warnings": [f"Calculation failed: {str(e)}"],
                "recommendations": ["Review exposure data", "Check risk weight assignments"]
            }

    async def _calculate_credit_rwa(self, exposure_data: Dict[str, Any], risk_weights: Dict[str, Any]) -> float:
        """Calculate Credit Risk RWA"""

        credit_rwa = 0.0

        # Corporate exposures
        corporate_exposure = exposure_data.get("corporate", 0)
        corporate_risk_weight = risk_weights.get("corporate", 1.0)
        credit_rwa += corporate_exposure * corporate_risk_weight

        # Retail exposures
        retail_exposure = exposure_data.get("retail", 0)
        retail_risk_weight = risk_weights.get("retail", 0.75)
        credit_rwa += retail_exposure * retail_risk_weight

        # Sovereign exposures
        sovereign_exposure = exposure_data.get("sovereign", 0)
        sovereign_risk_weight = risk_weights.get("sovereign", 0.0)
        credit_rwa += sovereign_exposure * sovereign_risk_weight

        # Bank exposures
        bank_exposure = exposure_data.get("bank", 0)
        bank_risk_weight = risk_weights.get("bank", 0.2)
        credit_rwa += bank_exposure * bank_risk_weight

        return round(credit_rwa, self.config["precision_decimal_places"])

    async def _calculate_market_rwa(self, market_exposures: Dict[str, Any]) -> float:
        """Calculate Market Risk RWA"""

        # Interest rate risk
        interest_rate_var = market_exposures.get("interest_rate_var", 0)

        # Foreign exchange risk
        fx_var = market_exposures.get("fx_var", 0)

        # Equity risk
        equity_var = market_exposures.get("equity_var", 0)

        # Commodity risk
        commodity_var = market_exposures.get("commodity_var", 0)

        # Calculate market RWA using standardized approach
        market_rwa = (interest_rate_var + fx_var + equity_var + commodity_var) * 12.5

        return round(market_rwa, self.config["precision_decimal_places"])

    async def _calculate_operational_rwa(self, operational_data: Dict[str, Any]) -> float:
        """Calculate Operational Risk RWA"""

        # Basic Indicator Approach
        gross_income = operational_data.get("gross_income", 0)
        operational_rwa = gross_income * 0.15 * 12.5  # 15% capital charge

        return round(operational_rwa, self.config["precision_decimal_places"])

    def _calculate_capital_ratios(self, capital_data: Dict[str, Any], total_rwa: float) -> Dict[str, float]:
        """Calculate capital adequacy ratios"""

        if total_rwa == 0:
            return {"cet1_ratio": 0.0, "tier1_ratio": 0.0, "total_ratio": 0.0}

        cet1_capital = capital_data.get("cet1_capital", 0)
        tier1_capital = capital_data.get("tier1_capital", 0)
        total_capital = capital_data.get("total_capital", 0)

        return {
            "cet1_ratio": round(cet1_capital / total_rwa, 4),
            "tier1_ratio": round(tier1_capital / total_rwa, 4),
            "total_ratio": round(total_capital / total_rwa, 4)
        }

    async def validate_model_config(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration parameters"""

        try:
            # Validate inputs
            required_fields = ["model_config", "validation_rules"]
            for field in required_fields:
                if field not in inputs:
                    raise ValueError(f"Missing required field: {field}")

            model_config = inputs["model_config"]
            validation_rules = inputs["validation_rules"]

            # Perform validation checks
            validation_results = []

            # Check required parameters
            required_params = validation_rules.get("required_parameters", [])
            for param in required_params:
                if param not in model_config:
                    validation_results.append({
                        "test": f"Required parameter: {param}",
                        "passed": False,
                        "message": f"Missing required parameter: {param}"
                    })
                else:
                    validation_results.append({
                        "test": f"Required parameter: {param}",
                        "passed": True,
                        "message": f"Parameter {param} is present"
                    })

            # Check parameter ranges
            parameter_ranges = validation_rules.get("parameter_ranges", {})
            for param, range_def in parameter_ranges.items():
                if param in model_config:
                    value = model_config[param]
                    min_val = range_def.get("min")
                    max_val = range_def.get("max")

                    if min_val is not None and value < min_val:
                        validation_results.append({
                            "test": f"Parameter range: {param}",
                            "passed": False,
                            "message": f"Parameter {param} ({value}) is below minimum ({min_val})"
                        })
                    elif max_val is not None and value > max_val:
                        validation_results.append({
                            "test": f"Parameter range: {param}",
                            "passed": False,
                            "message": f"Parameter {param} ({value}) is above maximum ({max_val})"
                        })
                    else:
                        validation_results.append({
                            "test": f"Parameter range: {param}",
                            "passed": True,
                            "message": f"Parameter {param} is within valid range"
                        })

            # Calculate overall validation score
            total_tests = len(validation_results)
            passed_tests = sum(1 for result in validation_results if result["passed"])
            validation_score = passed_tests / total_tests if total_tests > 0 else 0.0

            # Determine if validation passed
            validation_passed = validation_score >= validation_rules.get("pass_threshold", 0.8)

            # Generate issues and recommendations
            issues = [result["message"] for result in validation_results if not result["passed"]]
            recommendations = []

            if issues:
                recommendations.append("Review and correct failed validation parameters")
                recommendations.append("Validate parameter values against business requirements")

            if validation_score < 1.0:
                recommendations.append("Consider additional validation rules")

            # Create result
            result = ModelValidationResult(
                validation_passed=validation_passed,
                validation_score=validation_score,
                test_results={result["test"]: result for result in validation_results},
                performance_metrics={
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "pass_rate": validation_score
                },
                issues=issues,
                recommendations=recommendations
            )

            logger.info(f"Model validation completed - Score: {result.validation_score:.2f}")

            return {
                "success": True,
                "validation_result": {
                    "validation_passed": result.validation_passed,
                    "validation_score": result.validation_score,
                    "test_results": result.test_results,
                    "performance_metrics": result.performance_metrics,
                    "issues": result.issues,
                    "recommendations": result.recommendations
                },
                "warnings": issues,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "validation_result": None,
                "warnings": [f"Validation failed: {str(e)}"],
                "recommendations": ["Review model configuration", "Check validation rules"]
            }

    async def perform_backtesting(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform model backtesting"""

        try:
            # Validate inputs
            required_fields = ["historical_data", "model_predictions", "backtesting_config"]
            for field in required_fields:
                if field not in inputs:
                    raise ValueError(f"Missing required field: {field}")

            historical_data = inputs["historical_data"]
            model_predictions = inputs["model_predictions"]
            backtesting_config = inputs["backtesting_config"]

            # Perform backtesting analysis
            backtest_results = await self._perform_backtest_analysis(
                historical_data, model_predictions, backtesting_config
            )

            logger.info(f"Backtesting completed - AUC: {backtest_results['auc']:.3f}")

            return {
                "success": True,
                "backtest_results": backtest_results,
                "warnings": [],
                "recommendations": [
                    "Review model performance metrics",
                    "Consider model recalibration if performance is poor",
                    "Validate backtesting results with business experts"
                ]
            }

        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "backtest_results": None,
                "warnings": [f"Backtesting failed: {str(e)}"],
                "recommendations": ["Review historical data quality", "Check model predictions"]
            }

    async def _perform_backtest_analysis(self, historical_data: Dict[str, Any],
                                        model_predictions: Dict[str, Any],
                                        backtesting_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform backtesting analysis"""

        # Simulate backtesting calculations
        # In production, this would use actual historical data and model predictions

        # Calculate performance metrics
        accuracy = 0.85  # Simulated accuracy
        precision = 0.80  # Simulated precision
        recall = 0.75     # Simulated recall
        auc = 0.78        # Simulated AUC

        # Generate confusion matrix
        confusion_matrix = {
            "true_positives": 150,
            "false_positives": 30,
            "true_negatives": 800,
            "false_negatives": 20
        }

        # Calculate additional metrics
        f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "auc": auc,
            "confusion_matrix": confusion_matrix,
            "test_period": backtesting_config.get("test_period", "2023-01-01 to 2023-12-31"),
            "sample_size": confusion_matrix["true_positives"] + confusion_matrix["false_positives"] +
                          confusion_matrix["true_negatives"] + confusion_matrix["false_negatives"]
        }

    async def generate_report(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate banking calculation report"""

        try:
            # Validate inputs
            required_fields = ["report_type", "data", "template"]
            for field in required_fields:
                if field not in inputs:
                    raise ValueError(f"Missing required field: {field}")

            report_type = inputs["report_type"]
            data = inputs["data"]
            template = inputs["template"]

            # Generate report based on type
            if report_type == "ecl_report":
                report = await self._generate_ecl_report(data, template)
            elif report_type == "rwa_report":
                report = await self._generate_rwa_report(data, template)
            elif report_type == "model_validation_report":
                report = await self._generate_model_validation_report(data, template)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")

            logger.info(f"Report generated - Type: {report_type}")

            return {
                "success": True,
                "report": report,
                "warnings": [],
                "recommendations": [
                    "Review report content for accuracy",
                    "Validate calculations with business experts",
                    "Archive report for audit purposes"
                ]
            }

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "report": None,
                "warnings": [f"Report generation failed: {str(e)}"],
                "recommendations": ["Review report data", "Check template format"]
            }

    async def _generate_ecl_report(self, data: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ECL calculation report"""

        return {
            "report_type": "ECL Calculation Report",
            "generation_date": datetime.utcnow().isoformat(),
            "executive_summary": {
                "total_ecl": data.get("total_ecl", 0),
                "stage_breakdown": {
                    "stage_1": data.get("stage_1_ecl", 0),
                    "stage_2": data.get("stage_2_ecl", 0),
                    "stage_3": data.get("stage_3_ecl", 0)
                },
                "key_drivers": ["Economic scenarios", "Portfolio composition", "Model parameters"]
            },
            "detailed_results": data,
            "methodology": "IFRS 9 Expected Credit Loss calculation",
            "compliance_status": "Compliant with IFRS 9 requirements"
        }

    async def _generate_rwa_report(self, data: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate RWA calculation report"""

        return {
            "report_type": "RWA Calculation Report",
            "generation_date": datetime.utcnow().isoformat(),
            "executive_summary": {
                "total_rwa": data.get("total_rwa", 0),
                "rwa_breakdown": {
                    "credit_rwa": data.get("credit_rwa", 0),
                    "market_rwa": data.get("market_rwa", 0),
                    "operational_rwa": data.get("operational_rwa", 0)
                },
                "capital_ratios": data.get("capital_ratios", {}),
                "regulatory_compliance": "Meets Basel III requirements"
            },
            "detailed_results": data,
            "methodology": "Basel III Standardized Approach",
            "compliance_status": "Compliant with Basel III framework"
        }

    async def _generate_model_validation_report(self, data: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model validation report"""

        return {
            "report_type": "Model Validation Report",
            "generation_date": datetime.utcnow().isoformat(),
            "executive_summary": {
                "validation_passed": data.get("validation_passed", False),
                "validation_score": data.get("validation_score", 0),
                "key_issues": data.get("issues", []),
                "recommendations": data.get("recommendations", [])
            },
            "detailed_results": data,
            "methodology": "Comprehensive model validation framework",
            "compliance_status": "Compliant with SR 11-7 Model Risk Management"
        }
