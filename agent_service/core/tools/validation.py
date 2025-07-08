import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import re
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(str, Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    level: ValidationLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    overall_score: float
    validation_results: List[ValidationResult]
    summary: Dict[str, Any]

class ValidationToolkit:
    """Data validation and quality checking utilities"""

    def __init__(self):
        self.config = {
            "max_missing_ratio": 0.1,  # 10% max missing values
            "min_data_points": 100,
            "outlier_threshold": 3,  # Standard deviations
            "date_format": "%Y-%m-%d",
            "decimal_precision": 4
        }

    async def validate_portfolio_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate portfolio data for banking calculations"""

        try:
            # Validate inputs
            if "portfolio_data" not in inputs:
                raise ValueError("Missing portfolio_data in inputs")

            portfolio_data = inputs["portfolio_data"]
            validation_rules = inputs.get("validation_rules", {})

            # Perform validation checks
            validation_results = []

            # Data completeness check
            completeness_result = await self._check_data_completeness(portfolio_data)
            validation_results.extend(completeness_result)

            # Data consistency check
            consistency_result = await self._check_data_consistency(portfolio_data)
            validation_results.extend(consistency_result)

            # Data quality check
            quality_result = await self._check_data_quality(portfolio_data)
            validation_results.extend(quality_result)

            # Business logic validation
            business_result = await self._check_business_logic(portfolio_data, validation_rules)
            validation_results.extend(business_result)

            # Calculate overall validation score
            total_checks = len(validation_results)
            passed_checks = sum(1 for result in validation_results if result.passed)
            overall_score = passed_checks / total_checks if total_checks > 0 else 0.0

            # Generate report
            report = DataQualityReport(
                total_checks=total_checks,
                passed_checks=passed_checks,
                failed_checks=total_checks - passed_checks,
                warnings=sum(1 for result in validation_results if result.level == ValidationLevel.WARNING),
                overall_score=overall_score,
                validation_results=validation_results,
                summary=self._generate_validation_summary(validation_results)
            )

            logger.info(f"Portfolio data validation completed - Score: {overall_score:.2f}")

            return {
                "success": True,
                "validation_report": {
                    "total_checks": report.total_checks,
                    "passed_checks": report.passed_checks,
                    "failed_checks": report.failed_checks,
                    "warnings": report.warnings,
                    "overall_score": report.overall_score,
                    "validation_results": [
                        {
                            "check_name": result.check_name,
                            "passed": result.passed,
                            "level": result.level.value,
                            "message": result.message,
                            "details": result.details
                        } for result in report.validation_results
                    ],
                    "summary": report.summary
                },
                "warnings": [result.message for result in validation_results if result.level == ValidationLevel.WARNING],
                "recommendations": self._generate_recommendations(validation_results)
            }

        except Exception as e:
            logger.error(f"Portfolio data validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "validation_report": None,
                "warnings": [f"Validation failed: {str(e)}"],
                "recommendations": ["Review portfolio data structure", "Check validation rules"]
            }

    async def _check_data_completeness(self, portfolio_data: Dict[str, Any]) -> List[ValidationResult]:
        """Check data completeness"""

        results = []

        # Required fields check
        required_fields = ["account_id", "balance", "product_type", "origination_date"]

        if isinstance(portfolio_data, dict):
            # Single record validation
            for field in required_fields:
                if field not in portfolio_data or portfolio_data[field] is None:
                    results.append(ValidationResult(
                        check_name=f"Required field: {field}",
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"Missing required field: {field}",
                        details={"field": field, "value": portfolio_data.get(field)},
                        timestamp=datetime.utcnow()
                    ))
                else:
                    results.append(ValidationResult(
                        check_name=f"Required field: {field}",
                        passed=True,
                        level=ValidationLevel.INFO,
                        message=f"Field {field} is present",
                        details={"field": field, "value": portfolio_data.get(field)},
                        timestamp=datetime.utcnow()
                    ))

        elif isinstance(portfolio_data, list):
            # Multiple records validation
            total_records = len(portfolio_data)

            for field in required_fields:
                missing_count = 0
                for record in portfolio_data:
                    if field not in record or record[field] is None:
                        missing_count += 1

                missing_ratio = missing_count / total_records if total_records > 0 else 0

                if missing_ratio > self.config["max_missing_ratio"]:
                    results.append(ValidationResult(
                        check_name=f"Field completeness: {field}",
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"Field {field} has {missing_ratio:.1%} missing values (exceeds {self.config['max_missing_ratio']:.1%})",
                        details={"field": field, "missing_count": missing_count, "total_records": total_records, "missing_ratio": missing_ratio},
                        timestamp=datetime.utcnow()
                    ))
                elif missing_ratio > 0:
                    results.append(ValidationResult(
                        check_name=f"Field completeness: {field}",
                        passed=True,
                        level=ValidationLevel.WARNING,
                        message=f"Field {field} has {missing_ratio:.1%} missing values",
                        details={"field": field, "missing_count": missing_count, "total_records": total_records, "missing_ratio": missing_ratio},
                        timestamp=datetime.utcnow()
                    ))
                else:
                    results.append(ValidationResult(
                        check_name=f"Field completeness: {field}",
                        passed=True,
                        level=ValidationLevel.INFO,
                        message=f"Field {field} is complete",
                        details={"field": field, "missing_count": 0, "total_records": total_records},
                        timestamp=datetime.utcnow()
                    ))

        return results

    async def _check_data_consistency(self, portfolio_data: Dict[str, Any]) -> List[ValidationResult]:
        """Check data consistency"""

        results = []

        if isinstance(portfolio_data, list):
            # Check for duplicate account IDs
            account_ids = [record.get("account_id") for record in portfolio_data if record.get("account_id")]
            unique_ids = set(account_ids)

            if len(account_ids) != len(unique_ids):
                results.append(ValidationResult(
                    check_name="Account ID uniqueness",
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Found {len(account_ids) - len(unique_ids)} duplicate account IDs",
                    details={"total_ids": len(account_ids), "unique_ids": len(unique_ids)},
                    timestamp=datetime.utcnow()
                ))
            else:
                results.append(ValidationResult(
                    check_name="Account ID uniqueness",
                    passed=True,
                    level=ValidationLevel.INFO,
                    message="All account IDs are unique",
                    details={"total_ids": len(account_ids)},
                    timestamp=datetime.utcnow()
                ))

            # Check date consistency
            for i, record in enumerate(portfolio_data):
                if "origination_date" in record and "maturity_date" in record:
                    try:
                        orig_date = datetime.strptime(record["origination_date"], self.config["date_format"])
                        mat_date = datetime.strptime(record["maturity_date"], self.config["date_format"])

                        if orig_date >= mat_date:
                            results.append(ValidationResult(
                                check_name=f"Date consistency - Record {i}",
                                passed=False,
                                level=ValidationLevel.ERROR,
                                message=f"Origination date ({record['origination_date']}) is not before maturity date ({record['maturity_date']})",
                                details={"record_index": i, "origination_date": record["origination_date"], "maturity_date": record["maturity_date"]},
                                timestamp=datetime.utcnow()
                            ))
                    except ValueError as e:
                        results.append(ValidationResult(
                            check_name=f"Date format - Record {i}",
                            passed=False,
                            level=ValidationLevel.ERROR,
                            message=f"Invalid date format: {str(e)}",
                            details={"record_index": i, "error": str(e)},
                            timestamp=datetime.utcnow()
                        ))

        return results

    async def _check_data_quality(self, portfolio_data: Dict[str, Any]) -> List[ValidationResult]:
        """Check data quality"""

        results = []

        if isinstance(portfolio_data, list):
            # Check for outliers in numerical fields
            numerical_fields = ["balance", "interest_rate", "credit_score"]

            for field in numerical_fields:
                values = []
                for record in portfolio_data:
                    if field in record and record[field] is not None:
                        try:
                            values.append(float(record[field]))
                        except (ValueError, TypeError):
                            continue

                if len(values) >= self.config["min_data_points"]:
                    mean_val = np.mean(values)
                    std_val = np.std(values)

                    outliers = [v for v in values if abs(v - mean_val) > self.config["outlier_threshold"] * std_val]
                    outlier_ratio = len(outliers) / len(values)

                    if outlier_ratio > 0.05:  # More than 5% outliers
                        results.append(ValidationResult(
                            check_name=f"Outlier detection: {field}",
                            passed=False,
                            level=ValidationLevel.WARNING,
                            message=f"Field {field} has {outlier_ratio:.1%} outliers",
                            details={"field": field, "outlier_count": len(outliers), "total_values": len(values), "outlier_ratio": outlier_ratio},
                            timestamp=datetime.utcnow()
                        ))
                    else:
                        results.append(ValidationResult(
                            check_name=f"Outlier detection: {field}",
                            passed=True,
                            level=ValidationLevel.INFO,
                            message=f"Field {field} has acceptable outlier ratio ({outlier_ratio:.1%})",
                            details={"field": field, "outlier_count": len(outliers), "total_values": len(values)},
                            timestamp=datetime.utcnow()
                        ))

        return results

    async def _check_business_logic(self, portfolio_data: Dict[str, Any], validation_rules: Dict[str, Any]) -> List[ValidationResult]:
        """Check business logic validation"""

        results = []

        # Balance range validation
        balance_rules = validation_rules.get("balance_rules", {})
        min_balance = balance_rules.get("min_balance", 0)
        max_balance = balance_rules.get("max_balance", float('inf'))

        if isinstance(portfolio_data, list):
            invalid_balances = 0
            for record in portfolio_data:
                if "balance" in record and record["balance"] is not None:
                    try:
                        balance = float(record["balance"])
                        if balance < min_balance or balance > max_balance:
                            invalid_balances += 1
                    except (ValueError, TypeError):
                        invalid_balances += 1

            if invalid_balances > 0:
                results.append(ValidationResult(
                    check_name="Balance range validation",
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Found {invalid_balances} records with invalid balance ranges",
                    details={"invalid_count": invalid_balances, "min_balance": min_balance, "max_balance": max_balance},
                    timestamp=datetime.utcnow()
                ))
            else:
                results.append(ValidationResult(
                    check_name="Balance range validation",
                    passed=True,
                    level=ValidationLevel.INFO,
                    message="All balances are within valid ranges",
                    details={"min_balance": min_balance, "max_balance": max_balance},
                    timestamp=datetime.utcnow()
                ))

        # Product type validation
        valid_products = validation_rules.get("valid_product_types", [])
        if valid_products and isinstance(portfolio_data, list):
            invalid_products = 0
            for record in portfolio_data:
                if "product_type" in record and record["product_type"] not in valid_products:
                    invalid_products += 1

            if invalid_products > 0:
                results.append(ValidationResult(
                    check_name="Product type validation",
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Found {invalid_products} records with invalid product types",
                    details={"invalid_count": invalid_products, "valid_products": valid_products},
                    timestamp=datetime.utcnow()
                ))
            else:
                results.append(ValidationResult(
                    check_name="Product type validation",
                    passed=True,
                    level=ValidationLevel.INFO,
                    message="All product types are valid",
                    details={"valid_products": valid_products},
                    timestamp=datetime.utcnow()
                ))

        return results

    def _generate_validation_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate validation summary"""

        summary = {
            "total_checks": len(validation_results),
            "passed_checks": sum(1 for result in validation_results if result.passed),
            "failed_checks": sum(1 for result in validation_results if not result.passed),
            "errors": sum(1 for result in validation_results if result.level == ValidationLevel.ERROR),
            "warnings": sum(1 for result in validation_results if result.level == ValidationLevel.WARNING),
            "info": sum(1 for result in validation_results if result.level == ValidationLevel.INFO),
            "categories": {}
        }

        # Group by category
        for result in validation_results:
            category = result.check_name.split(":")[0] if ":" in result.check_name else "General"
            if category not in summary["categories"]:
                summary["categories"][category] = {"total": 0, "passed": 0, "failed": 0}

            summary["categories"][category]["total"] += 1
            if result.passed:
                summary["categories"][category]["passed"] += 1
            else:
                summary["categories"][category]["failed"] += 1

        return summary

    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""

        recommendations = []

        # Check for common issues
        error_results = [result for result in validation_results if result.level == ValidationLevel.ERROR]
        warning_results = [result for result in validation_results if result.level == ValidationLevel.WARNING]

        if error_results:
            recommendations.append("Address critical data quality issues identified in validation")

            # Missing data recommendations
            missing_data_errors = [r for r in error_results if "missing" in r.message.lower()]
            if missing_data_errors:
                recommendations.append("Implement data completeness checks in source systems")

        if warning_results:
            recommendations.append("Review and address data quality warnings")

            # Outlier recommendations
            outlier_warnings = [r for r in warning_results if "outlier" in r.message.lower()]
            if outlier_warnings:
                recommendations.append("Investigate outlier values for potential data quality issues")

        # General recommendations
        recommendations.extend([
            "Implement automated data validation in ETL processes",
            "Set up data quality monitoring dashboards",
            "Establish data governance policies for ongoing quality assurance"
        ])

        return recommendations

    async def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs for banking calculations"""

        try:
            calculation_type = inputs.get("calculation_type", "unknown")
            calculation_inputs = inputs.get("calculation_inputs", {})

            validation_results = []

            # Type-specific validation
            if calculation_type == "ecl_calculation":
                validation_results.extend(await self._validate_ecl_inputs(calculation_inputs))
            elif calculation_type == "rwa_calculation":
                validation_results.extend(await self._validate_rwa_inputs(calculation_inputs))
            elif calculation_type == "model_validation":
                validation_results.extend(await self._validate_model_inputs(calculation_inputs))

            # Common validation
            validation_results.extend(await self._validate_common_inputs(calculation_inputs))

            # Calculate validation score
            total_checks = len(validation_results)
            passed_checks = sum(1 for result in validation_results if result.passed)
            validation_score = passed_checks / total_checks if total_checks > 0 else 0.0

            logger.info(f"Calculation input validation completed - Score: {validation_score:.2f}")

            return {
                "success": True,
                "validation_score": validation_score,
                "validation_results": [
                    {
                        "check_name": result.check_name,
                        "passed": result.passed,
                        "level": result.level.value,
                        "message": result.message,
                        "details": result.details
                    } for result in validation_results
                ],
                "warnings": [result.message for result in validation_results if result.level == ValidationLevel.WARNING],
                "recommendations": self._generate_recommendations(validation_results)
            }

        except Exception as e:
            logger.error(f"Calculation input validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "validation_score": 0.0,
                "validation_results": [],
                "warnings": [f"Validation failed: {str(e)}"],
                "recommendations": ["Review calculation inputs", "Check input data structure"]
            }

    async def _validate_ecl_inputs(self, inputs: Dict[str, Any]) -> List[ValidationResult]:
        """Validate ECL calculation inputs"""

        results = []

        # Required fields for ECL calculation
        required_fields = ["portfolio_data", "pd_curves", "lgd_estimates", "ead_estimates"]

        for field in required_fields:
            if field not in inputs:
                results.append(ValidationResult(
                    check_name=f"ECL input: {field}",
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing required ECL input: {field}",
                    details={"field": field},
                    timestamp=datetime.utcnow()
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"ECL input: {field}",
                    passed=True,
                    level=ValidationLevel.INFO,
                    message=f"ECL input {field} is present",
                    details={"field": field},
                    timestamp=datetime.utcnow()
                ))

        # Validate PD curves
        if "pd_curves" in inputs:
            pd_curves = inputs["pd_curves"]
            if "12m_pd" in pd_curves:
                pd_12m = pd_curves["12m_pd"]
                if not (0 <= pd_12m <= 1):
                    results.append(ValidationResult(
                        check_name="PD validation: 12m_pd",
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"12-month PD ({pd_12m}) must be between 0 and 1",
                        details={"pd_12m": pd_12m},
                        timestamp=datetime.utcnow()
                    ))

        return results

    async def _validate_rwa_inputs(self, inputs: Dict[str, Any]) -> List[ValidationResult]:
        """Validate RWA calculation inputs"""

        results = []

        # Required fields for RWA calculation
        required_fields = ["exposure_data", "risk_weights", "capital_data"]

        for field in required_fields:
            if field not in inputs:
                results.append(ValidationResult(
                    check_name=f"RWA input: {field}",
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing required RWA input: {field}",
                    details={"field": field},
                    timestamp=datetime.utcnow()
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"RWA input: {field}",
                    passed=True,
                    level=ValidationLevel.INFO,
                    message=f"RWA input {field} is present",
                    details={"field": field},
                    timestamp=datetime.utcnow()
                ))

        # Validate risk weights
        if "risk_weights" in inputs:
            risk_weights = inputs["risk_weights"]
            for category, weight in risk_weights.items():
                if not (0 <= weight <= 10):  # Reasonable range for risk weights
                    results.append(ValidationResult(
                        check_name=f"Risk weight: {category}",
                        passed=False,
                        level=ValidationLevel.WARNING,
                        message=f"Risk weight for {category} ({weight}) seems unusually high",
                        details={"category": category, "weight": weight},
                        timestamp=datetime.utcnow()
                    ))

        return results

    async def _validate_model_inputs(self, inputs: Dict[str, Any]) -> List[ValidationResult]:
        """Validate model validation inputs"""

        results = []

        # Required fields for model validation
        required_fields = ["model_config", "validation_rules"]

        for field in required_fields:
            if field not in inputs:
                results.append(ValidationResult(
                    check_name=f"Model validation input: {field}",
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing required model validation input: {field}",
                    details={"field": field},
                    timestamp=datetime.utcnow()
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"Model validation input: {field}",
                    passed=True,
                    level=ValidationLevel.INFO,
                    message=f"Model validation input {field} is present",
                    details={"field": field},
                    timestamp=datetime.utcnow()
                ))

        return results

    async def _validate_common_inputs(self, inputs: Dict[str, Any]) -> List[ValidationResult]:
        """Validate common calculation inputs"""

        results = []

        # Check for empty inputs
        if not inputs:
            results.append(ValidationResult(
                check_name="Input completeness",
                passed=False,
                level=ValidationLevel.ERROR,
                message="No calculation inputs provided",
                details={"inputs": inputs},
                timestamp=datetime.utcnow()
            ))
        else:
            results.append(ValidationResult(
                check_name="Input completeness",
                passed=True,
                level=ValidationLevel.INFO,
                message="Calculation inputs provided",
                details={"input_count": len(inputs)},
                timestamp=datetime.utcnow()
            ))

        return results

    async def validate_regulatory_compliance(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regulatory compliance"""

        try:
            regulation_type = inputs.get("regulation_type", "unknown")
            data_to_validate = inputs.get("data", {})

            validation_results = []

            # Regulation-specific validation
            if regulation_type == "ifrs9":
                validation_results.extend(await self._validate_ifrs9_compliance(data_to_validate))
            elif regulation_type == "basel3":
                validation_results.extend(await self._validate_basel3_compliance(data_to_validate))
            elif regulation_type == "sr_11_7":
                validation_results.extend(await self._validate_sr11_7_compliance(data_to_validate))

            # Calculate compliance score
            total_checks = len(validation_results)
            passed_checks = sum(1 for result in validation_results if result.passed)
            compliance_score = passed_checks / total_checks if total_checks > 0 else 0.0

            logger.info(f"Regulatory compliance validation completed - Score: {compliance_score:.2f}")

            return {
                "success": True,
                "compliance_score": compliance_score,
                "regulation_type": regulation_type,
                "validation_results": [
                    {
                        "check_name": result.check_name,
                        "passed": result.passed,
                        "level": result.level.value,
                        "message": result.message,
                        "details": result.details
                    } for result in validation_results
                ],
                "warnings": [result.message for result in validation_results if result.level == ValidationLevel.WARNING],
                "recommendations": self._generate_recommendations(validation_results)
            }

        except Exception as e:
            logger.error(f"Regulatory compliance validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "compliance_score": 0.0,
                "validation_results": [],
                "warnings": [f"Compliance validation failed: {str(e)}"],
                "recommendations": ["Review regulatory requirements", "Check compliance data"]
            }

    async def _validate_ifrs9_compliance(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate IFRS 9 compliance"""

        results = []

        # Check for required IFRS 9 elements
        ifrs9_requirements = ["staging_criteria", "forward_looking_information", "lifetime_ecl_calculation"]

        for requirement in ifrs9_requirements:
            if requirement not in data:
                results.append(ValidationResult(
                    check_name=f"IFRS 9: {requirement}",
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing IFRS 9 requirement: {requirement}",
                    details={"requirement": requirement},
                    timestamp=datetime.utcnow()
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"IFRS 9: {requirement}",
                    passed=True,
                    level=ValidationLevel.INFO,
                    message=f"IFRS 9 requirement {requirement} is satisfied",
                    details={"requirement": requirement},
                    timestamp=datetime.utcnow()
                ))

        return results

    async def _validate_basel3_compliance(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate Basel III compliance"""

        results = []

        # Check capital ratios
        capital_ratios = data.get("capital_ratios", {})

        # CET1 ratio minimum 4.5%
        cet1_ratio = capital_ratios.get("cet1_ratio", 0)
        if cet1_ratio < 0.045:
            results.append(ValidationResult(
                check_name="Basel III: CET1 ratio",
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"CET1 ratio ({cet1_ratio:.2%}) below minimum requirement (4.5%)",
                details={"cet1_ratio": cet1_ratio, "minimum": 0.045},
                timestamp=datetime.utcnow()
            ))
        else:
            results.append(ValidationResult(
                check_name="Basel III: CET1 ratio",
                passed=True,
                level=ValidationLevel.INFO,
                message=f"CET1 ratio ({cet1_ratio:.2%}) meets minimum requirement",
                details={"cet1_ratio": cet1_ratio, "minimum": 0.045},
                timestamp=datetime.utcnow()
            ))

        # Tier 1 ratio minimum 6%
        tier1_ratio = capital_ratios.get("tier1_ratio", 0)
        if tier1_ratio < 0.06:
            results.append(ValidationResult(
                check_name="Basel III: Tier 1 ratio",
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Tier 1 ratio ({tier1_ratio:.2%}) below minimum requirement (6%)",
                details={"tier1_ratio": tier1_ratio, "minimum": 0.06},
                timestamp=datetime.utcnow()
            ))
        else:
            results.append(ValidationResult(
                check_name="Basel III: Tier 1 ratio",
                passed=True,
                level=ValidationLevel.INFO,
                message=f"Tier 1 ratio ({tier1_ratio:.2%}) meets minimum requirement",
                details={"tier1_ratio": tier1_ratio, "minimum": 0.06},
                timestamp=datetime.utcnow()
            ))

        return results

    async def _validate_sr11_7_compliance(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate SR 11-7 Model Risk Management compliance"""

        results = []

        # Check for model governance elements
        governance_elements = ["model_documentation", "independent_validation", "ongoing_monitoring"]

        for element in governance_elements:
            if element not in data:
                results.append(ValidationResult(
                    check_name=f"SR 11-7: {element}",
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing SR 11-7 requirement: {element}",
                    details={"element": element},
                    timestamp=datetime.utcnow()
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"SR 11-7: {element}",
                    passed=True,
                    level=ValidationLevel.INFO,
                    message=f"SR 11-7 requirement {element} is satisfied",
                    details={"element": element},
                    timestamp=datetime.utcnow()
                ))

        return results
