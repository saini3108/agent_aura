"""
Intelligent Analyst Agent with Real LLM Analysis
Performs comprehensive credit risk data analysis using AI
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List
from datetime import datetime
import sys
from pathlib import Path

# Add shared path for imports
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir / "shared"))

from llm_provider import llm_manager, truncate_text, count_tokens

class IntelligentAnalystAgent:
    """AI-powered analyst for credit risk data analysis"""
    
    def __init__(self):
        self.agent_name = "intelligent_analyst"
        self.llm_manager = llm_manager
        self.max_context_tokens = 8000
        
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform intelligent analysis using LLM capabilities
        """
        start_time = datetime.now()
        
        try:
            # Extract data and context
            data = context.get('data')
            files = context.get('files', {})
            risk_thresholds = context.get('risk_thresholds', {})
            
            if data is None or data.empty:
                return self._generate_error_response("No data provided for analysis")
            
            # Perform comprehensive analysis
            analysis_results = {
                "data_overview": self._analyze_data_overview(data),
                "statistical_analysis": self._perform_statistical_analysis(data),
                "risk_indicators": self._identify_risk_indicators(data),
                "feature_insights": self._analyze_features(data),
                "data_quality": self._assess_data_quality(data),
                "ai_insights": self._generate_ai_insights(data, risk_thresholds),
                "recommendations": self._generate_recommendations(data),
                "execution_metadata": {
                    "agent": self.agent_name,
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat(),
                    "data_shape": data.shape,
                    "llm_provider": self.llm_manager.default_provider if self.llm_manager.providers else "none"
                }
            }
            
            return {
                "status": "completed",
                "analysis": analysis_results,
                "agent_type": "intelligent_analyst",
                "llm_enhanced": True
            }
            
        except Exception as e:
            return self._generate_error_response(f"Analysis failed: {str(e)}")
    
    def _analyze_data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data overview"""
        overview = {
            "total_records": len(data),
            "total_features": len(data.columns),
            "feature_types": {},
            "missing_data": {},
            "data_ranges": {},
            "target_distribution": {}
        }
        
        # Analyze feature types
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                overview["feature_types"][col] = "numeric"
                overview["data_ranges"][col] = {
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std())
                }
            else:
                overview["feature_types"][col] = "categorical"
                overview["data_ranges"][col] = {
                    "unique_values": int(data[col].nunique()),
                    "most_common": str(data[col].mode().iloc[0]) if not data[col].empty else "N/A"
                }
            
            # Missing data analysis
            missing_count = data[col].isnull().sum()
            overview["missing_data"][col] = {
                "count": int(missing_count),
                "percentage": float(missing_count / len(data) * 100)
            }
        
        # Target variable analysis (assuming 'default' column)
        if 'default' in data.columns:
            target_dist = data['default'].value_counts()
            overview["target_distribution"] = {
                "default_rate": float(data['default'].mean()),
                "non_default_count": int(target_dist.get(0, 0)),
                "default_count": int(target_dist.get(1, 0))
            }
        
        return overview
    
    def _perform_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on the data"""
        stats = {
            "correlations": {},
            "distributions": {},
            "outliers": {},
            "relationships": {}
        }
        
        # Correlation analysis for numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            
            # Find high correlations
            high_corr_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": float(corr_val)
                        })
            
            stats["correlations"]["high_correlations"] = high_corr_pairs
            
            # Target correlations (if default column exists)
            if 'default' in data.columns:
                target_corrs = []
                for col in numeric_cols:
                    if col != 'default':
                        corr_val = data[col].corr(data['default'])
                        target_corrs.append({
                            "feature": col,
                            "correlation_with_default": float(corr_val) if not np.isnan(corr_val) else 0.0
                        })
                stats["correlations"]["target_correlations"] = sorted(
                    target_corrs, 
                    key=lambda x: abs(x["correlation_with_default"]), 
                    reverse=True
                )
        
        # Outlier detection using IQR method
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            stats["outliers"][col] = {
                "count": len(outliers),
                "percentage": float(len(outliers) / len(data) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
        
        return stats
    
    def _identify_risk_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify key risk indicators in the data"""
        risk_indicators = {
            "high_risk_segments": [],
            "risk_factors": [],
            "portfolio_metrics": {}
        }
        
        if 'default' in data.columns:
            # Analyze risk by different segments
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if data[col].nunique() < 10:  # Only analyze categorical variables with reasonable number of categories
                    segment_risk = data.groupby(col)['default'].agg(['count', 'mean']).reset_index()
                    segment_risk.columns = [col, 'count', 'default_rate']
                    
                    high_risk_segments = segment_risk[segment_risk['default_rate'] > data['default'].mean() * 1.5]
                    
                    for _, segment in high_risk_segments.iterrows():
                        risk_indicators["high_risk_segments"].append({
                            "segment": col,
                            "value": str(segment[col]),
                            "default_rate": float(segment['default_rate']),
                            "population": int(segment['count']),
                            "risk_level": "high" if segment['default_rate'] > 0.3 else "medium"
                        })
            
            # Analyze numeric risk factors
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'default':
                    correlation = data[col].corr(data['default'])
                    if abs(correlation) > 0.2:  # Significant correlation
                        risk_indicators["risk_factors"].append({
                            "factor": col,
                            "correlation": float(correlation),
                            "direction": "increases_risk" if correlation > 0 else "decreases_risk",
                            "strength": "strong" if abs(correlation) > 0.5 else "moderate"
                        })
            
            # Portfolio-level metrics
            risk_indicators["portfolio_metrics"] = {
                "overall_default_rate": float(data['default'].mean()),
                "total_exposure": int(len(data)),
                "risk_concentration": self._calculate_risk_concentration(data)
            }
        
        return risk_indicators
    
    def _calculate_risk_concentration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk concentration metrics"""
        concentration = {}
        
        # Geographic concentration (if available)
        if any(col for col in data.columns if 'region' in col.lower() or 'state' in col.lower()):
            geo_cols = [col for col in data.columns if 'region' in col.lower() or 'state' in col.lower()]
            for col in geo_cols:
                top_regions = data[col].value_counts().head(5)
                concentration[f"{col}_concentration"] = {
                    "top_5_regions_percentage": float(top_regions.sum() / len(data) * 100),
                    "herfindahl_index": float(sum((data[col].value_counts() / len(data)) ** 2))
                }
        
        # Industry concentration (if available)
        industry_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['industry', 'sector', 'business'])]
        for col in industry_cols:
            if data[col].nunique() > 1:
                top_industries = data[col].value_counts().head(5)
                concentration[f"{col}_concentration"] = {
                    "top_5_industries_percentage": float(top_industries.sum() / len(data) * 100),
                    "diversification_index": float(1 - sum((data[col].value_counts() / len(data)) ** 2))
                }
        
        return concentration
    
    def _analyze_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual features for modeling insights"""
        feature_insights = {
            "predictive_power": [],
            "data_quality_issues": [],
            "feature_engineering_opportunities": []
        }
        
        for col in data.columns:
            if col == 'default':
                continue
                
            insight = {
                "feature": col,
                "type": str(data[col].dtype),
                "missing_percentage": float(data[col].isnull().sum() / len(data) * 100),
                "unique_values": int(data[col].nunique()),
                "quality_score": 0.0
            }
            
            # Calculate quality score
            quality_score = 100.0
            if insight["missing_percentage"] > 10:
                quality_score -= insight["missing_percentage"]
            if insight["unique_values"] <= 1:
                quality_score -= 50
            if data[col].dtype == 'object' and insight["unique_values"] > len(data) * 0.9:
                quality_score -= 30  # High cardinality categorical
            
            insight["quality_score"] = max(0.0, quality_score)
            
            # Predictive power analysis
            if 'default' in data.columns and data[col].dtype in ['int64', 'float64']:
                correlation = abs(data[col].corr(data['default']))
                if not np.isnan(correlation):
                    insight["correlation_with_target"] = float(correlation)
                    if correlation > 0.3:
                        insight["predictive_power"] = "high"
                    elif correlation > 0.1:
                        insight["predictive_power"] = "medium"
                    else:
                        insight["predictive_power"] = "low"
            
            feature_insights["predictive_power"].append(insight)
            
            # Identify data quality issues
            if insight["missing_percentage"] > 20:
                feature_insights["data_quality_issues"].append({
                    "feature": col,
                    "issue": "high_missing_data",
                    "severity": "high" if insight["missing_percentage"] > 50 else "medium",
                    "description": f"{insight['missing_percentage']:.1f}% missing values"
                })
        
        return feature_insights
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        quality_assessment = {
            "overall_score": 0.0,
            "completeness": 0.0,
            "consistency": 0.0,
            "validity": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Completeness assessment
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells * 100
        quality_assessment["completeness"] = float(completeness)
        
        # Consistency checks
        consistency_issues = 0
        
        # Check for negative values in fields that should be positive
        positive_fields = ['income', 'loan_amount', 'credit_score', 'age']
        for field in positive_fields:
            if field in data.columns and data[field].dtype in ['int64', 'float64']:
                negative_count = (data[field] < 0).sum()
                if negative_count > 0:
                    consistency_issues += negative_count
                    quality_assessment["issues"].append({
                        "type": "negative_values",
                        "field": field,
                        "count": int(negative_count),
                        "severity": "high"
                    })
        
        # Check for unrealistic values
        if 'age' in data.columns:
            unrealistic_age = ((data['age'] < 18) | (data['age'] > 100)).sum()
            if unrealistic_age > 0:
                consistency_issues += unrealistic_age
                quality_assessment["issues"].append({
                    "type": "unrealistic_values",
                    "field": "age",
                    "count": int(unrealistic_age),
                    "severity": "medium"
                })
        
        if 'credit_score' in data.columns:
            invalid_credit_score = ((data['credit_score'] < 300) | (data['credit_score'] > 850)).sum()
            if invalid_credit_score > 0:
                consistency_issues += invalid_credit_score
                quality_assessment["issues"].append({
                    "type": "out_of_range",
                    "field": "credit_score",
                    "count": int(invalid_credit_score),
                    "severity": "high"
                })
        
        consistency = max(0, 100 - (consistency_issues / len(data) * 100))
        quality_assessment["consistency"] = float(consistency)
        
        # Validity assessment (basic format checks)
        validity_score = 100.0
        
        # Check for duplicates
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            validity_score -= (duplicate_count / len(data) * 100)
            quality_assessment["issues"].append({
                "type": "duplicates",
                "count": int(duplicate_count),
                "severity": "medium"
            })
        
        quality_assessment["validity"] = float(validity_score)
        
        # Overall score (weighted average)
        overall_score = (
            quality_assessment["completeness"] * 0.4 +
            quality_assessment["consistency"] * 0.4 +
            quality_assessment["validity"] * 0.2
        )
        quality_assessment["overall_score"] = float(overall_score)
        
        # Generate recommendations
        if quality_assessment["completeness"] < 90:
            quality_assessment["recommendations"].append("Address missing data through imputation or collection")
        if quality_assessment["consistency"] < 85:
            quality_assessment["recommendations"].append("Investigate and correct data consistency issues")
        if len(quality_assessment["issues"]) > 0:
            quality_assessment["recommendations"].append("Review flagged data quality issues before modeling")
        
        return quality_assessment
    
    def _generate_ai_insights(self, data: pd.DataFrame, risk_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights using LLM"""
        if not self.llm_manager.get_available_providers():
            return {
                "insights": ["LLM provider not available - unable to generate AI insights"],
                "recommendations": ["Configure LLM API key to enable AI-powered analysis"],
                "risk_assessment": "Unable to perform AI risk assessment"
            }
        
        try:
            # Prepare data summary for LLM
            data_summary = self._prepare_data_summary_for_llm(data)
            
            # Create comprehensive prompt for analysis
            prompt = self._create_analysis_prompt(data_summary, risk_thresholds)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert credit risk analyst with deep knowledge of banking regulations, statistical modeling, and risk management. Provide detailed, actionable insights based on the data provided."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Get LLM response
            response = self.llm_manager.chat_completion(
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=2000
            )
            
            # Parse and structure the response
            ai_insights = self._parse_llm_insights(response.content)
            
            # Add metadata
            ai_insights["llm_metadata"] = {
                "provider": response.provider,
                "model": response.model,
                "tokens_used": response.usage["total_tokens"],
                "timestamp": datetime.now().isoformat()
            }
            
            return ai_insights
            
        except Exception as e:
            return {
                "insights": [f"AI analysis failed: {str(e)}"],
                "recommendations": ["Check LLM configuration and try again"],
                "risk_assessment": "AI risk assessment unavailable"
            }
    
    def _prepare_data_summary_for_llm(self, data: pd.DataFrame) -> str:
        """Prepare a concise data summary for LLM analysis"""
        summary_parts = []
        
        # Basic data info
        summary_parts.append(f"Dataset: {len(data)} records, {len(data.columns)} features")
        
        # Target variable info
        if 'default' in data.columns:
            default_rate = data['default'].mean()
            summary_parts.append(f"Default rate: {default_rate:.2%}")
        
        # Key statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limit to top 5 numeric columns
            if col != 'default':
                stats = data[col].describe()
                summary_parts.append(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        
        # Categorical columns info
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols[:3]:  # Limit to top 3 categorical columns
            unique_count = data[col].nunique()
            top_value = data[col].mode().iloc[0] if not data[col].empty else "N/A"
            summary_parts.append(f"{col}: {unique_count} unique values, most common: {top_value}")
        
        # Missing data
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            missing_cols = missing_data[missing_data > 0]
            summary_parts.append(f"Missing data: {', '.join([f'{col}({count})' for col, count in missing_cols.items()])}")
        
        return "; ".join(summary_parts)
    
    def _create_analysis_prompt(self, data_summary: str, risk_thresholds: Dict[str, Any]) -> str:
        """Create comprehensive analysis prompt for LLM"""
        prompt = f"""
As a senior credit risk analyst, analyze the following dataset for a credit risk model validation:

DATA SUMMARY:
{data_summary}

RISK THRESHOLDS:
{json.dumps(risk_thresholds, indent=2) if risk_thresholds else "No thresholds provided"}

Please provide a comprehensive analysis covering:

1. KEY INSIGHTS: What are the most important patterns and relationships in this data?

2. RISK FACTORS: Identify the primary risk drivers and their potential impact on default probability.

3. DATA QUALITY ASSESSMENT: Evaluate the data quality and identify any concerns for modeling.

4. REGULATORY CONSIDERATIONS: What regulatory aspects (Basel III, IFRS 9, SR 11-7) should be considered?

5. MODEL DEVELOPMENT RECOMMENDATIONS: What modeling approaches would be most suitable?

6. VALIDATION PRIORITIES: What aspects should receive priority attention during validation?

7. BUSINESS IMPLICATIONS: What are the key business risks and opportunities identified?

Please be specific, actionable, and focus on practical recommendations for model validation and risk management.
"""
        
        # Ensure prompt doesn't exceed token limits
        return truncate_text(prompt, max_tokens=self.max_context_tokens // 2)
    
    def _parse_llm_insights(self, llm_response: str) -> Dict[str, Any]:
        """Parse and structure LLM response into actionable insights"""
        insights = {
            "key_insights": [],
            "risk_factors": [],
            "data_quality_notes": [],
            "regulatory_considerations": [],
            "recommendations": [],
            "business_implications": [],
            "overall_risk_assessment": "medium",
            "confidence_level": "medium"
        }
        
        # Simple parsing - in production, you might use more sophisticated NLP
        lines = llm_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if "KEY INSIGHTS" in line.upper():
                current_section = "key_insights"
            elif "RISK FACTORS" in line.upper():
                current_section = "risk_factors"
            elif "DATA QUALITY" in line.upper():
                current_section = "data_quality_notes"
            elif "REGULATORY" in line.upper():
                current_section = "regulatory_considerations"
            elif "RECOMMENDATIONS" in line.upper() or "MODEL DEVELOPMENT" in line.upper():
                current_section = "recommendations"
            elif "BUSINESS IMPLICATIONS" in line.upper():
                current_section = "business_implications"
            elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                # Bullet point - add to current section
                clean_line = line.lstrip('-•* ').strip()
                if current_section and clean_line:
                    insights[current_section].append(clean_line)
            elif current_section and len(line) > 20:  # Regular paragraph text
                insights[current_section].append(line)
        
        # Extract overall risk assessment
        response_lower = llm_response.lower()
        if any(word in response_lower for word in ['high risk', 'significant risk', 'elevated risk']):
            insights["overall_risk_assessment"] = "high"
        elif any(word in response_lower for word in ['low risk', 'minimal risk', 'acceptable risk']):
            insights["overall_risk_assessment"] = "low"
        
        # If no specific insights were parsed, add the full response
        if not any(insights[key] for key in ["key_insights", "risk_factors", "recommendations"]):
            insights["key_insights"] = [llm_response]
        
        return insights
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Generate specific recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        missing_percentage = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
        if missing_percentage > 10:
            recommendations.append(f"Address missing data ({missing_percentage:.1f}% of all values) before model development")
        
        # Sample size recommendations
        if len(data) < 1000:
            recommendations.append("Consider increasing sample size for more robust model development")
        elif len(data) > 100000:
            recommendations.append("Large dataset - consider sampling strategies for efficient model development")
        
        # Feature engineering recommendations
        numeric_features = len(data.select_dtypes(include=[np.number]).columns)
        if numeric_features < 5:
            recommendations.append("Limited numeric features - consider feature engineering to create additional predictive variables")
        
        # Target variable recommendations
        if 'default' in data.columns:
            default_rate = data['default'].mean()
            if default_rate < 0.05:
                recommendations.append("Low default rate - consider stratified sampling and specialized techniques for imbalanced data")
            elif default_rate > 0.3:
                recommendations.append("High default rate - review data collection criteria and consider portfolio risk management")
        
        # Validation recommendations
        recommendations.extend([
            "Implement cross-validation with temporal splits for time-sensitive credit data",
            "Monitor model performance across different customer segments",
            "Establish ongoing monitoring for data drift and model degradation",
            "Document all assumptions and limitations for regulatory compliance"
        ])
        
        return recommendations
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate standardized error response"""
        return {
            "status": "error",
            "error_message": error_message,
            "analysis": {
                "data_overview": {},
                "execution_metadata": {
                    "agent": self.agent_name,
                    "timestamp": datetime.now().isoformat(),
                    "error": True
                }
            },
            "agent_type": "intelligent_analyst",
            "llm_enhanced": False
        }