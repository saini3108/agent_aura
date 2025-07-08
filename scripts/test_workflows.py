#!/usr/bin/env python3
"""
Test workflows for the Banking AI Platform
Run this to verify all workflow types work correctly
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000/api/v1"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_model_validation():
    """Test model validation workflow"""
    print("\n🧠 Testing model validation workflow...")

    workflow_data = {
        "workflow_type": "model_validation",
        "inputs": {
            "model_name": "credit_risk_model_v1",
            "model_configuration": {
                "algorithm": "logistic_regression",
                "features": ["credit_score", "income", "debt_ratio"],
                "target": "default_probability"
            }
        },
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4"
        }
    }

    response = requests.post(f"{BASE_URL}/workflows/start", json=workflow_data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        workflow_id = result["workflow_id"]
        print(f"✅ Workflow started: {workflow_id}")
        return workflow_id
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_ecl_calculation():
    """Test ECL calculation workflow"""
    print("\n💰 Testing ECL calculation workflow...")

    workflow_data = {
        "workflow_type": "ecl_calculation",
        "inputs": {
            "portfolio_data": [
                {
                    "account_id": "ACC001",
                    "balance": 50000,
                    "product_type": "personal_loan",
                    "origination_date": "2023-01-15",
                    "maturity_date": "2028-01-15",
                    "interest_rate": 5.5,
                    "credit_score": 720
                },
                {
                    "account_id": "ACC002",
                    "balance": 25000,
                    "product_type": "credit_card",
                    "origination_date": "2023-06-01",
                    "credit_score": 650
                }
            ],
            "scenarios": ["base", "stress", "optimistic"]
        }
    }

    response = requests.post(f"{BASE_URL}/workflows/start", json=workflow_data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        workflow_id = result["workflow_id"]
        print(f"✅ Workflow started: {workflow_id}")
        return workflow_id
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_rwa_calculation():
    """Test RWA calculation workflow"""
    print("\n📊 Testing RWA calculation workflow...")

    workflow_data = {
        "workflow_type": "rwa_calculation",
        "inputs": {
            "exposure_data": [
                {
                    "exposure_id": "EXP001",
                    "counterparty": "Corporate_A",
                    "exposure_amount": 1000000,
                    "asset_class": "corporate",
                    "rating": "BBB",
                    "maturity": 3.5,
                    "collateral_type": "none"
                },
                {
                    "exposure_id": "EXP002",
                    "counterparty": "Bank_B",
                    "exposure_amount": 500000,
                    "asset_class": "bank",
                    "rating": "A",
                    "maturity": 1.0
                }
            ]
        }
    }

    response = requests.post(f"{BASE_URL}/workflows/start", json=workflow_data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        workflow_id = result["workflow_id"]
        print(f"✅ Workflow started: {workflow_id}")
        return workflow_id
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_reporting():
    """Test reporting workflow"""
    print("\n📋 Testing reporting workflow...")

    workflow_data = {
        "workflow_type": "reporting",
        "inputs": {
            "report_type": "risk_summary",
            "data_sources": ["portfolio_data", "market_data", "regulatory_data"],
            "template": "monthly_risk_report",
            "filters": {
                "date_range": "2024-01-01 to 2024-12-31",
                "business_unit": "retail_banking"
            }
        }
    }

    response = requests.post(f"{BASE_URL}/workflows/start", json=workflow_data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        workflow_id = result["workflow_id"]
        print(f"✅ Workflow started: {workflow_id}")
        return workflow_id
    else:
        print(f"❌ Error: {response.text}")
        return None

def check_workflow_status(workflow_id: str):
    """Check workflow status"""
    if not workflow_id:
        return

    print(f"\n📄 Checking status for workflow: {workflow_id}")
    response = requests.get(f"{BASE_URL}/workflows/{workflow_id}/status")

    if response.status_code == 200:
        status = response.json()
        print(f"Status: {status.get('status', 'unknown')}")
        print(f"Current step: {status.get('current_step', 0)}")
    else:
        print(f"❌ Error getting status: {response.text}")

def validate_workflow_request():
    """Test workflow validation endpoint"""
    print("\n✅ Testing workflow validation...")

    validation_data = {
        "workflow_type": "model_validation",
        "inputs": {
            "model_name": "test_model",
            "model_configuration": {"algorithm": "test"}
        }
    }

    response = requests.post(f"{BASE_URL}/workflows/validate", json=validation_data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Valid: {result['valid']}")
        if result.get('errors'):
            print(f"Errors: {result['errors']}")
        if result.get('warnings'):
            print(f"Warnings: {result['warnings']}")
    else:
        print(f"❌ Error: {response.text}")

def main():
    """Run all tests"""
    print("🚀 Banking AI Platform - Workflow Tests")
    print("=" * 50)

    # Test basic connectivity
    if not test_health_check():
        print("❌ Health check failed - server not responding")
        return

    # Test workflow validation
    validate_workflow_request()

    # Test all workflow types
    workflow_ids = []

    workflow_ids.append(test_model_validation())
    workflow_ids.append(test_ecl_calculation())
    workflow_ids.append(test_rwa_calculation())
    workflow_ids.append(test_reporting())

    # Wait a moment for workflows to process
    print("\n⏳ Waiting for workflows to process...")
    time.sleep(2)

    # Check status of all workflows
    for workflow_id in workflow_ids:
        if workflow_id:
            check_workflow_status(workflow_id)

    print("\n✅ Test suite completed!")
    print("\n📖 Next steps:")
    print("1. Check the API documentation at http://localhost:8000/docs")
    print("2. Monitor logs for detailed workflow execution")
    print("3. Use the endpoints to interact with running workflows")

if __name__ == "__main__":
    main()
