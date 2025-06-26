import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime
import re

class DocumentationAgent:
    """Agent responsible for reviewing compliance documentation"""
    
    def __init__(self):
        self.name = "Documentation Agent"
        self.description = "Reviews model documentation for compliance and completeness"
        
        # Define required documentation sections
        self.required_sections = [
            'model_purpose',
            'data_sources',
            'methodology',
            'validation_results',
            'limitations',
            'assumptions',
            'approval_process'
        ]
        
        # Define compliance keywords
        self.compliance_keywords = {
            'basel': ['basel', 'capital', 'regulatory', 'pillar'],
            'ifrs9': ['ifrs', 'ifrs9', 'expected credit loss', 'ecl', 'staging'],
            'model_risk': ['model risk', 'validation', 'independent', 'review'],
            'governance': ['governance', 'approval', 'committee', 'oversight'],
            'backtesting': ['backtesting', 'performance', 'monitoring', 'calibration']
        }
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for the documentation agent
        
        Args:
            context: Dictionary containing data, files, and previous outputs
            
        Returns:
            Dictionary containing documentation review results
        """
        try:
            files = context.get('files', {})
            previous_outputs = context.get('previous_outputs', {})
            
            documentation_results = {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'completed',
                'review_results': {}
            }
            
            if not files:
                documentation_results['status'] = 'warning'
                documentation_results['message'] = 'No documentation files provided for review'
                documentation_results['review_results'] = self._generate_no_docs_review()
                return documentation_results
            
            # Review each uploaded file
            file_reviews = {}
            for filename, file_info in files.items():
                file_review = self._review_file(filename, file_info)
                file_reviews[filename] = file_review
            
            # Generate overall documentation assessment
            overall_assessment = self._generate_overall_assessment(file_reviews)
            
            documentation_results['review_results'] = {
                'file_reviews': file_reviews,
                'overall_assessment': overall_assessment,
                'compliance_checklist': self._generate_compliance_checklist(file_reviews),
                'recommendations': self._generate_recommendations(file_reviews, overall_assessment)
            }
            
            return documentation_results
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'status': 'error',
                'error': str(e)
            }
    
    def _review_file(self, filename: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Review a single documentation file"""
        file_review = {
            'filename': filename,
            'file_type': self._get_file_type(filename),
            'file_size': file_info.get('size', 0),
            'upload_date': file_info.get('uploaded_at', ''),
            'content_analysis': {}
        }
        
        # Analyze filename for content hints
        filename_analysis = self._analyze_filename(filename)
        file_review['content_analysis']['filename_analysis'] = filename_analysis
        
        # Since we can't read file contents in this implementation,
        # we'll provide analysis based on filename and file type
        content_assessment = self._assess_content_by_filename(filename)
        file_review['content_analysis']['content_assessment'] = content_assessment
        
        # Compliance relevance
        compliance_relevance = self._assess_compliance_relevance(filename)
        file_review['compliance_relevance'] = compliance_relevance
        
        return file_review
    
    def _get_file_type(self, filename: str) -> str:
        """Determine file type from filename"""
        if '.' in filename:
            return filename.split('.')[-1].lower()
        return 'unknown'
    
    def _analyze_filename(self, filename: str) -> Dict[str, Any]:
        """Analyze filename for content indicators"""
        filename_lower = filename.lower()
        
        analysis = {
            'document_type': 'unknown',
            'content_indicators': [],
            'potential_sections': []
        }
        
        # Document type indicators
        if any(keyword in filename_lower for keyword in ['validation', 'validate']):
            analysis['document_type'] = 'validation_report'
            analysis['content_indicators'].append('validation_focused')
        
        if any(keyword in filename_lower for keyword in ['model', 'methodology']):
            analysis['document_type'] = 'model_documentation'
            analysis['content_indicators'].append('model_methodology')
        
        if any(keyword in filename_lower for keyword in ['policy', 'procedure']):
            analysis['document_type'] = 'policy_document'
            analysis['content_indicators'].append('governance_policy')
        
        if any(keyword in filename_lower for keyword in ['risk', 'assessment']):
            analysis['document_type'] = 'risk_assessment'
            analysis['content_indicators'].append('risk_management')
        
        # Regulatory indicators
        if any(keyword in filename_lower for keyword in ['basel', 'pillar']):
            analysis['content_indicators'].append('basel_compliance')
        
        if any(keyword in filename_lower for keyword in ['ifrs', 'ecl']):
            analysis['content_indicators'].append('ifrs9_compliance')
        
        # Technical indicators
        if any(keyword in filename_lower for keyword in ['technical', 'spec', 'specification']):
            analysis['content_indicators'].append('technical_specifications')
        
        return analysis
    
    def _assess_content_by_filename(self, filename: str) -> Dict[str, Any]:
        """Assess likely content based on filename"""
        filename_lower = filename.lower()
        
        assessment = {
            'likely_sections': [],
            'compliance_areas': [],
            'completeness_score': 0.0,
            'quality_indicators': []
        }
        
        # Predict likely sections based on filename
        if 'validation' in filename_lower:
            assessment['likely_sections'].extend([
                'validation_results', 'performance_metrics', 'backtesting'
            ])
            assessment['completeness_score'] += 0.3
        
        if 'model' in filename_lower:
            assessment['likely_sections'].extend([
                'methodology', 'assumptions', 'limitations'
            ])
            assessment['completeness_score'] += 0.2
        
        if 'data' in filename_lower:
            assessment['likely_sections'].extend([
                'data_sources', 'data_quality'
            ])
            assessment['completeness_score'] += 0.1
        
        if 'risk' in filename_lower:
            assessment['likely_sections'].extend([
                'risk_assessment', 'model_risk'
            ])
            assessment['completeness_score'] += 0.1
        
        # Quality indicators based on filename
        if any(keyword in filename_lower for keyword in ['final', 'approved', 'signed']):
            assessment['quality_indicators'].append('appears_finalized')
            assessment['completeness_score'] += 0.1
        
        if any(keyword in filename_lower for keyword in ['draft', 'temp', 'wip']):
            assessment['quality_indicators'].append('draft_document')
            assessment['completeness_score'] -= 0.2
        
        # Compliance areas
        for area, keywords in self.compliance_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                assessment['compliance_areas'].append(area)
        
        # Normalize completeness score
        assessment['completeness_score'] = max(0.0, min(1.0, assessment['completeness_score']))
        
        return assessment
    
    def _assess_compliance_relevance(self, filename: str) -> Dict[str, Any]:
        """Assess compliance relevance of the document"""
        filename_lower = filename.lower()
        
        relevance = {
            'regulatory_frameworks': [],
            'compliance_score': 0.0,
            'required_for_compliance': False
        }
        
        # Check for regulatory frameworks
        if any(keyword in filename_lower for keyword in self.compliance_keywords['basel']):
            relevance['regulatory_frameworks'].append('Basel III')
            relevance['compliance_score'] += 0.3
        
        if any(keyword in filename_lower for keyword in self.compliance_keywords['ifrs9']):
            relevance['regulatory_frameworks'].append('IFRS 9')
            relevance['compliance_score'] += 0.3
        
        if any(keyword in filename_lower for keyword in self.compliance_keywords['model_risk']):
            relevance['regulatory_frameworks'].append('Model Risk Management')
            relevance['compliance_score'] += 0.2
        
        # Check if document is required for compliance
        if any(keyword in filename_lower for keyword in ['validation', 'approval', 'policy']):
            relevance['required_for_compliance'] = True
            relevance['compliance_score'] += 0.2
        
        return relevance
    
    def _generate_overall_assessment(self, file_reviews: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall documentation assessment"""
        assessment = {
            'total_files': len(file_reviews),
            'documentation_completeness': 0.0,
            'compliance_coverage': [],
            'missing_sections': [],
            'quality_score': 0.0,
            'status': 'incomplete'
        }
        
        if not file_reviews:
            assessment['status'] = 'no_documentation'
            return assessment
        
        # Calculate overall scores
        total_completeness = 0.0
        total_quality = 0.0
        all_compliance_areas = set()
        
        for filename, review in file_reviews.items():
            content_assessment = review.get('content_analysis', {}).get('content_assessment', {})
            total_completeness += content_assessment.get('completeness_score', 0.0)
            
            # Quality assessment based on file characteristics
            quality_score = 0.5  # Base score
            if 'appears_finalized' in content_assessment.get('quality_indicators', []):
                quality_score += 0.3
            if 'draft_document' in content_assessment.get('quality_indicators', []):
                quality_score -= 0.2
            
            total_quality += max(0.0, quality_score)
            
            # Collect compliance areas
            compliance_areas = review.get('compliance_relevance', {}).get('regulatory_frameworks', [])
            all_compliance_areas.update(compliance_areas)
        
        # Calculate averages
        assessment['documentation_completeness'] = total_completeness / len(file_reviews)
        assessment['quality_score'] = total_quality / len(file_reviews)
        assessment['compliance_coverage'] = list(all_compliance_areas)
        
        # Determine status
        if assessment['documentation_completeness'] >= 0.8 and assessment['quality_score'] >= 0.7:
            assessment['status'] = 'comprehensive'
        elif assessment['documentation_completeness'] >= 0.6:
            assessment['status'] = 'adequate'
        elif assessment['documentation_completeness'] >= 0.3:
            assessment['status'] = 'partial'
        else:
            assessment['status'] = 'insufficient'
        
        # Identify missing sections
        covered_sections = set()
        for filename, review in file_reviews.items():
            content_assessment = review.get('content_analysis', {}).get('content_assessment', {})
            covered_sections.update(content_assessment.get('likely_sections', []))
        
        missing_sections = [section for section in self.required_sections 
                          if section not in covered_sections]
        assessment['missing_sections'] = missing_sections
        
        return assessment
    
    def _generate_compliance_checklist(self, file_reviews: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate compliance checklist"""
        checklist = {
            'regulatory_compliance': {},
            'documentation_standards': {},
            'governance_requirements': {}
        }
        
        # Regulatory compliance checks
        regulatory_frameworks = set()
        for filename, review in file_reviews.items():
            frameworks = review.get('compliance_relevance', {}).get('regulatory_frameworks', [])
            regulatory_frameworks.update(frameworks)
        
        checklist['regulatory_compliance'] = {
            'basel_iii_coverage': 'Basel III' in regulatory_frameworks,
            'ifrs9_coverage': 'IFRS 9' in regulatory_frameworks,
            'model_risk_coverage': 'Model Risk Management' in regulatory_frameworks
        }
        
        # Documentation standards
        has_validation_doc = any('validation' in review.get('content_analysis', {}).get('filename_analysis', {}).get('document_type', '')
                               for review in file_reviews.values())
        has_methodology_doc = any('model' in review.get('content_analysis', {}).get('filename_analysis', {}).get('document_type', '')
                                for review in file_reviews.values())
        
        checklist['documentation_standards'] = {
            'validation_documentation': has_validation_doc,
            'methodology_documentation': has_methodology_doc,
            'approval_documentation': False  # Would need content analysis to determine
        }
        
        # Governance requirements
        checklist['governance_requirements'] = {
            'independent_validation': has_validation_doc,
            'approval_process_documented': False,  # Would need content analysis
            'regular_review_process': False  # Would need content analysis
        }
        
        return checklist
    
    def _generate_recommendations(self, file_reviews: Dict[str, Dict[str, Any]], 
                                 overall_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on documentation review"""
        recommendations = []
        
        # Overall status recommendations
        status = overall_assessment.get('status', 'unknown')
        if status == 'insufficient':
            recommendations.append("Provide comprehensive model documentation before proceeding with validation")
        elif status == 'partial':
            recommendations.append("Enhance documentation coverage to meet regulatory requirements")
        
        # Missing sections recommendations
        missing_sections = overall_assessment.get('missing_sections', [])
        if missing_sections:
            recommendations.append(f"Provide documentation for missing sections: {', '.join(missing_sections)}")
        
        # Compliance recommendations
        compliance_coverage = overall_assessment.get('compliance_coverage', [])
        if 'Basel III' not in compliance_coverage:
            recommendations.append("Include Basel III compliance documentation")
        if 'IFRS 9' not in compliance_coverage:
            recommendations.append("Include IFRS 9 compliance documentation")
        
        # Quality recommendations
        quality_score = overall_assessment.get('quality_score', 0.0)
        if quality_score < 0.7:
            recommendations.append("Improve documentation quality and ensure all documents are finalized")
        
        # File-specific recommendations
        draft_files = []
        for filename, review in file_reviews.items():
            content_assessment = review.get('content_analysis', {}).get('content_assessment', {})
            if 'draft_document' in content_assessment.get('quality_indicators', []):
                draft_files.append(filename)
        
        if draft_files:
            recommendations.append(f"Finalize draft documents: {', '.join(draft_files)}")
        
        return recommendations
    
    def _generate_no_docs_review(self) -> Dict[str, Any]:
        """Generate review results when no documentation is provided"""
        return {
            'file_reviews': {},
            'overall_assessment': {
                'total_files': 0,
                'documentation_completeness': 0.0,
                'compliance_coverage': [],
                'missing_sections': self.required_sections,
                'quality_score': 0.0,
                'status': 'no_documentation'
            },
            'compliance_checklist': {
                'regulatory_compliance': {
                    'basel_iii_coverage': False,
                    'ifrs9_coverage': False,
                    'model_risk_coverage': False
                },
                'documentation_standards': {
                    'validation_documentation': False,
                    'methodology_documentation': False,
                    'approval_documentation': False
                },
                'governance_requirements': {
                    'independent_validation': False,
                    'approval_process_documented': False,
                    'regular_review_process': False
                }
            },
            'recommendations': [
                "Upload comprehensive model documentation",
                "Provide validation methodology documentation",
                "Include regulatory compliance documentation",
                "Ensure all required sections are covered"
            ]
        }
