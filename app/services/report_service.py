import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from PIL import Image as PILImage

from app.core.config import settings
from app.services.visualization_service import VisualizationService

logger = logging.getLogger(__name__)

class ReportService:
    def __init__(self):
        self.report_prompt = self._create_report_prompt()
        self.visualization_service = VisualizationService()
        try:
            self.llm_endpoint = HuggingFaceEndpoint(
                repo_id=settings.MODEL_NAME,
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_ACCESS_TOKEN,
                temperature=0.7,
                max_new_tokens=1024,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            self.chat_model = ChatHuggingFace(llm=self.llm_endpoint)
            logger.info(f"ReportService initialized successfully with model: {settings.MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize ReportService: {e}")
            self.chat_model = None
    
    def _create_report_prompt(self) -> PromptTemplate:

        template = """You are a senior radiologist specializing in neuro-oncology. Generate a comprehensive, well-structured clinical report based on the brain tumor segmentation analysis.

CLINICAL DATA:
{features_json}

PATIENT INFO:
{patient_info}

Generate a detailed medical report with clear sections, proper headings, and clinical formatting. Use the following EXACT structure:

**EXECUTIVE SUMMARY**
Provide a 2-3 sentence overview of the key findings and clinical significance.

**TUMOR MORPHOLOGY AND LOCATION**
• Location: [Describe hemisphere and anatomical region]
• Size Classification: [Based on volume measurements]
• Maximum Diameter: [Include measurement in mm]
• Anatomical Considerations: [Clinical relevance of location]

**QUANTITATIVE ANALYSIS**
• Total Tumor Volume: [volume] cm³
• Tumor Core Volume: [volume] cm³ 
• Enhancing Component: [volume] cm³ ([percentage]%)
• Necrotic Component: [volume] cm³ ([percentage]%)
• Edematous Component: [volume] cm³ ([percentage]%)

**ENHANCEMENT CHARACTERISTICS**
• Enhancement Pattern: [pattern description]
• Enhancement Intensity: Mean [value], Maximum [value]
• Clinical Significance: [interpretation of enhancement]

**TISSUE COMPOSITION ANALYSIS**
Present findings in a structured table format:
- Enhancing Tissue: [Present/Absent] - [Clinical interpretation]
- Necrotic Core: [Present/Absent] - [Clinical interpretation] 
- Peritumoral Edema: [Present/Absent] - [Clinical interpretation]

**CLINICAL ASSESSMENT**
• Tumor Grade Indicators: [Based on imaging features]
• Differential Diagnosis: [Likely tumor types based on characteristics]
• Prognosis Indicators: [Relevant imaging markers]

**RECOMMENDATIONS**
1. Immediate Actions: [Clinical priorities]
2. Additional Imaging: [If needed]
3. Multidisciplinary Review: [Team involvement recommendations]
4. Follow-up Protocol: [Monitoring schedule]
5. Treatment Considerations: [Relevant for imaging findings]

**TECHNICAL NOTES**
• Image Quality: Adequate for diagnostic interpretation
• Segmentation Confidence: High automated detection accuracy
• Limitations: Standard limitations of MRI-based analysis

Report Generated: {report_date}
System: AI-Assisted Brain Tumor Analysis Platform

Use proper medical terminology and ensure each section is clearly formatted with bullet points and structured information."""
        
        return PromptTemplate(
            input_variables=["features_json", "patient_info", "report_date"],
            template=template
        )
    
    def _generate_ai_report(self, prompt: str) -> str:

        try:
            if self.chat_model is None:
                logger.warning("Chat model not initialized, using enhanced fallback")
                return self._generate_enhanced_fallback_report()

            response = self.chat_model.invoke(prompt)

            if hasattr(response, 'content'):
                report_text = response.content.strip()
            else:
                report_text = str(response).strip()
            
            report_text = self._post_process_report(report_text)
            logger.info("AI repsponce post processing been started")
            
            return report_text
                
        except Exception as e:
            logger.error(f"Error generating AI report: {e}")
            return self._generate_enhanced_fallback_report()
    
    def _post_process_report(self, report: str) -> str:
        report = report.replace('#', '').replace('*', '')
        report = report.replace('```', '').replace('---', '')
        sections = [
            "EXECUTIVE SUMMARY", "TUMOR MORPHOLOGY AND LOCATION", 
            "QUANTITATIVE ANALYSIS", "ENHANCEMENT CHARACTERISTICS",
            "TISSUE COMPOSITION ANALYSIS", "CLINICAL ASSESSMENT",
            "RECOMMENDATIONS", "TECHNICAL NOTES"
        ]
        
        for section in sections:
            report = report.replace(section, f"\n**{section}**\n")
        import re
        report = re.sub(r'\n\s*\n', '\n\n', report)
        report = report.strip()
        
        return report
    
    def _generate_enhanced_fallback_report(self) -> str:
        return """**EXECUTIVE SUMMARY** <<<<<<<<<FallBack Report>>>>>>>>>>>
Automated brain tumor segmentation analysis has been completed successfully. The analysis provides comprehensive quantitative assessment of tumor components including enhancing regions, necrotic areas, and peritumoral edema with precise volumetric measurements.

**TUMOR MORPHOLOGY AND LOCATION**
• Location: Detailed anatomical localization based on centroid analysis
• Size Classification: Determined by total tumor volume measurements  
• Maximum Diameter: Calculated from 3D morphological analysis
• Anatomical Considerations: Location-specific clinical implications assessed

**QUANTITATIVE ANALYSIS**
• Total Tumor Volume: Comprehensive 3D volumetric assessment completed
• Tumor Core Volume: Active tumor tissue quantification performed
• Enhancing Component: Contrast-enhancing regions identified and measured
• Necrotic Component: Non-viable tissue areas quantified  
• Edematous Component: Peritumoral swelling extent determined

**ENHANCEMENT CHARACTERISTICS**
• Enhancement Pattern: Systematic analysis of contrast uptake patterns
• Enhancement Intensity: Statistical analysis of signal intensities
• Clinical Significance: Enhancement characteristics correlated with tumor biology

**TISSUE COMPOSITION ANALYSIS**
Comprehensive tissue characterization completed:
- Enhancing Tissue: Automated detection and classification performed
- Necrotic Core: Central non-enhancing regions identified
- Peritumoral Edema: Surrounding tissue involvement assessed

**CLINICAL ASSESSMENT**
• Tumor Grade Indicators: Imaging features analyzed for grade assessment
• Differential Diagnosis: Pattern recognition applied for tumor classification
• Prognosis Indicators: Quantitative markers evaluated for outcome prediction

**RECOMMENDATIONS**
1. Clinical correlation with patient symptoms and neurological examination
2. Multidisciplinary team review for comprehensive treatment planning
3. Follow-up imaging protocol based on tumor characteristics
4. Consider advanced imaging techniques for additional characterization
5. Monitor quantitative changes in follow-up studies

**TECHNICAL NOTES**
• Image Quality: Analysis performed on high-quality multi-parametric MRI data
• Segmentation Confidence: Advanced AI algorithms provide reliable tissue classification
• Limitations: Results should be interpreted within appropriate clinical context

This analysis provides objective quantitative data to support clinical decision-making and should be integrated with comprehensive patient evaluation."""
    
    def generate_report(self, features: Dict[str, Any], 
                       patient_info: Optional[Dict[str, str]] = None,
                       task_id: str = "",
                       file_paths: Optional[Dict[str, Path]] = None,
                       segmentation_path: Optional[Path] = None) -> Dict[str, Any]:

        try:
            features_formatted = self._format_features_for_report(features)
            patient_info_str = self._format_patient_info(patient_info or {})
            report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            
            formatted_prompt = self.report_prompt.format(
                features_json=features_formatted,
                patient_info=patient_info_str,
                report_date=report_date
            )
            
            logger.info(f"Generating structured AI report for task {task_id}")

            report_text = self._generate_ai_report(formatted_prompt)

            visualizations = {}
            if file_paths and segmentation_path:
                visualizations = self.visualization_service.create_all_modality_overlays(
                    file_paths, segmentation_path
                )

                visualizations['3d_volume'] = self.visualization_service.create_3d_volume_visualization(
                    segmentation_path
                )
        
            report_data = {
                "report_text": report_text,
                "features": features,
                "patient_info": patient_info or {},
                "visualizations": visualizations,
                "generated_at": datetime.now().isoformat(),
                "task_id": task_id,
                "model_used": "Qwen/Qwen3-Coder-30B-A3B-Instruct" if self.chat_model else "enhanced_fallback"
            }
            
            logger.info(f"Comprehensive AI report generated successfully for task {task_id}")
            return report_data
            
        except Exception as e:
            logger.error(f"Report generation failed for task {task_id}: {e}")
            return {
                "report_text": self._generate_enhanced_fallback_report(),
                "features": features,
                "patient_info": patient_info or {},
                "visualizations": {},
                "generated_at": datetime.now().isoformat(),
                "task_id": task_id,
                "model_used": "enhanced_fallback",
                "error": str(e)
            }
    
    def _format_features_for_report(self, features: Dict[str, Any]) -> str:
        formatted = f"""
CASE ID: {features.get('case_id', 'N/A')}
IMAGING PARAMETERS: {features.get('voxel_spacing_mm', 'N/A')} voxel spacing

VOLUMETRIC MEASUREMENTS:
- Whole Tumor Volume: {features.get('whole_tumor_volume_cm3', 0):.2f} cm³
- Tumor Core Volume: {features.get('tumor_core_volume_cm3', 0):.2f} cm³
- Enhancing Volume: {features.get('enhancing_volume_cm3', 0):.2f} cm³
- Necrotic Volume: {features.get('necrotic_volume_cm3', 0):.2f} cm³
- Edema Volume: {features.get('edema_volume_cm3', 0):.2f} cm³

DIMENSIONAL ANALYSIS:
- Maximum Tumor Diameter: {features.get('whole_tumor_diameter_mm', 0):.1f} mm
- Tumor Core Diameter: {features.get('tumor_core_diameter_mm', 0):.1f} mm
- Enhancing Diameter: {features.get('enhancing_diameter_mm', 0):.1f} mm

TISSUE COMPOSITION:
- Enhancement Percentage: {features.get('enhancing_percentage', 0):.1f}%
- Necrotic Percentage: {features.get('necrotic_percentage', 0):.1f}%
- Edema Percentage: {features.get('edema_percentage', 0):.1f}%

ANATOMICAL LOCATION:
- Hemisphere: {features.get('hemisphere', 'N/A')}
- Regional Location: {features.get('anatomical_location', 'N/A')}
- Centroid Coordinates: {features.get('centroid_coordinates', 'N/A')}

ENHANCEMENT CHARACTERISTICS:
- Mean Enhancement Intensity: {features.get('enhancement_mean_intensity', 0):.2f}
- Maximum Enhancement Intensity: {features.get('enhancement_max_intensity', 0):.2f}
- Enhancement Present: {features.get('has_enhancement', 'N/A')}

TISSUE MARKERS:
- Necrosis Present: {features.get('has_necrosis', 'N/A')}
- Edema Present: {features.get('has_edema', 'N/A')}
- Size Category: {features.get('tumor_size_category', 'N/A')}
- Enhancement Pattern: {features.get('enhancement_pattern', 'N/A')}
- Necrosis Extent: {features.get('necrosis_extent', 'N/A')}
        """
        return formatted.strip()
    
    def _format_patient_info(self, patient_info: Dict[str, str]) -> str:
        if not any(patient_info.values()):
            return "Patient information not provided"
        
        formatted_info = []
        field_mapping = {
            'patient_id': 'Patient ID',
            'patient_age': 'Age',
            'patient_gender': 'Gender', 
            'referring_physician': 'Referring Physician'
        }
        
        for key, label in field_mapping.items():
            value = patient_info.get(key)
            if value:
                formatted_info.append(f"{label}: {value}")
        
        return "\n".join(formatted_info) if formatted_info else "Patient information not provided"
    
    def _create_enhanced_visualization_charts(self, features: Dict[str, Any]) -> Dict[str, str]:
        charts = {}
        
        try:
            plt.style.use('default')
            sns.set_palette("Set2")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            volumes = [
                features['enhancing_volume_cm3'],
                features['necrotic_volume_cm3'],
                features['edema_volume_cm3']
            ]
            labels = ['Enhancing', 'Necrotic', 'Edema']
            colors = ['#e74c3c', '#f39c12', '#27ae60']

            non_zero_data = [(vol, label, color) for vol, label, color in zip(volumes, labels, colors) if vol > 0]
            if non_zero_data:
                volumes, labels, colors = zip(*non_zero_data)
                wedges, texts, autotexts = ax1.pie(volumes, labels=labels, autopct='%1.1f%%', 
                                                  colors=colors, startangle=90, explode=[0.1]*len(volumes))
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            ax1.set_title('Tumor Component Distribution', fontsize=14, fontweight='bold')
            components = ['Whole Tumor', 'Tumor Core', 'Enhancing', 'Necrotic', 'Edema']
            values = [
                features['whole_tumor_volume_cm3'],
                features['tumor_core_volume_cm3'], 
                features['enhancing_volume_cm3'],
                features['necrotic_volume_cm3'],
                features['edema_volume_cm3']
            ]
            
            bars = ax2.bar(components, values, color=['#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#27ae60'])
            ax2.set_ylabel('Volume (cm³)', fontsize=12)
            ax2.set_title('Component Volume Comparison', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)

            for bar, value in zip(bars, values):
                if value > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            charts['volume_analysis'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            enhancement_data = {
                'Enhancement %': features['enhancing_percentage'],
                'Necrosis %': features['necrotic_percentage'], 
                'Edema %': features['edema_percentage']
            }
            
            bars = ax1.bar(enhancement_data.keys(), enhancement_data.values(),
                          color=['#e74c3c', '#f39c12', '#27ae60'])
            ax1.set_ylabel('Percentage (%)', fontsize=12)
            ax1.set_title('Tissue Composition Percentages', fontsize=14, fontweight='bold')
            ax1.set_ylim(0, 100)
            
            for bar, value in zip(bars, enhancement_data.values()):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            size_categories = ['Small\n(<1 cm³)', 'Medium\n(1-5 cm³)', 'Large\n(5-15 cm³)', 'Very Large\n(>15 cm³)']
            current_volume = features['whole_tumor_volume_cm3']

            if current_volume < 1:
                current_cat = 0
            elif current_volume < 5:
                current_cat = 1
            elif current_volume < 15:
                current_cat = 2
            else:
                current_cat = 3
            
            colors = ['lightgray'] * 4
            colors[current_cat] = '#3498db'
            
            bars = ax2.bar(size_categories, [1, 1, 1, 1], color=colors, alpha=0.7)
            bars[current_cat].set_alpha(1.0)
            
            ax2.set_ylabel('Category Scale', fontsize=12)
            ax2.set_title(f'Tumor Size Classification\n(Current: {current_volume:.2f} cm³)', 
                         fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1.2)
            ax2.set_yticks([])

            ax2.text(current_cat, 1.1, f'{current_volume:.2f} cm³',
                    ha='center', va='bottom', fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            charts['classification_analysis'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            summary_data = [
                ['Parameter', 'Value', 'Clinical Significance'],
                ['Total Volume', f"{features['whole_tumor_volume_cm3']:.2f} cm³", features['tumor_size_category']],
                ['Maximum Diameter', f"{features['whole_tumor_diameter_mm']:.1f} mm", 'Surgical planning reference'],
                ['Enhancement', f"{features['enhancing_percentage']:.1f}%", features['enhancement_pattern']],
                ['Necrosis', f"{features['necrotic_percentage']:.1f}%", features['necrosis_extent']],
                ['Location', f"{features['hemisphere']} {features['anatomical_location']}", 'Functional considerations'],
                ['Enhancement Present', features['has_enhancement'], 'Blood-brain barrier disruption'],
                ['Necrosis Present', features['has_necrosis'], 'Tissue viability indicator'],
                ['Edema Present', features['has_edema'], 'Peritumoral involvement']
            ]
            
            table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                           cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            for i in range(len(summary_data[0])):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(1, len(summary_data)):
                for j in range(len(summary_data[0])):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f8f9fa')
                    table[(i, j)].set_text_props(wrap=True)
            
            ax.set_title('Clinical Summary Table', fontsize=16, fontweight='bold', pad=20)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            charts['clinical_summary'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating enhanced visualizations: {e}")
        
        return charts
    
    def generate_pdf_report(self, report_data: Dict[str, Any], output_path: Path) -> Path:

        try:
            doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                  rightMargin=0.75*inch, leftMargin=0.75*inch,
                                  topMargin=1*inch, bottomMargin=1*inch)
            styles = getSampleStyleSheet()
            story = []
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=20,
                spaceAfter=30,
                textColor=colors.HexColor('#2c3e50'),
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            subtitle_style = ParagraphStyle(
                'SubTitle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=20,
                textColor=colors.HexColor('#34495e'),
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=16,
                textColor=colors.HexColor('#2c3e50'),
                fontName='Helvetica-Bold'
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                alignment=TA_JUSTIFY,
                leftIndent=0.2*inch
            )
            
            story.append(Paragraph("BRAIN TUMOR ANALYSIS REPORT", title_style))
            story.append(Paragraph("AI-Powered Segmentation and Clinical Assessment", subtitle_style))
            story.append(Spacer(1, 0.5*inch))

            if report_data.get('patient_info'):
                story.append(Paragraph("Patient Information", heading_style))
                patient_data = [['Field', 'Value']]
                patient_data.append(['Report Date', report_data.get('generated_at', 'N/A')])
                patient_data.append(['Case ID', report_data.get('features', {}).get('case_id', 'N/A')])
                
                for key, value in report_data['patient_info'].items():
                    if value:
                        patient_data.append([key.replace('_', ' ').title(), str(value)])
                
                patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
                patient_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
                ]))
                story.append(patient_table)
                story.append(Spacer(1, 0.3*inch))
            
            story.append(PageBreak())

            story.append(Paragraph("AI-GENERATED CLINICAL REPORT", title_style))
            story.append(Spacer(1, 0.2*inch))

            report_text = report_data.get('report_text', '')
            sections = report_text.split('**')
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue

                if i % 2 == 1: 
                    story.append(Paragraph(section.strip(), heading_style))
                else:  
                    lines = section.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            if line.startswith('•') or line.startswith('-'):
                                story.append(Paragraph(line, body_style))
                            elif line.startswith(('1.', '2.', '3.', '4.', '5.')):
                                story.append(Paragraph(line, body_style))
                            elif ':' in line and len(line.split(':')) == 2:
                                story.append(Paragraph(line, body_style))
                            else:
                                story.append(Paragraph(line, body_style))
                    story.append(Spacer(1, 0.1*inch))
            
            story.append(PageBreak())

            story.append(Paragraph("SEGMENTATION VISUALIZATIONS", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            visualizations = report_data.get('visualizations', {})
            for modality, viz_data in visualizations.items():
                if viz_data and modality != '3d_volume':
                    story.append(Paragraph(f"{modality.upper()} Segmentation Overlay", heading_style))
                    try:
                        img_data = base64.b64decode(viz_data)
                        img_buffer = io.BytesIO(img_data)
                        img = Image(img_buffer, width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        logger.error(f"Error adding visualization {modality}: {e}")
            
            if '3d_volume' in visualizations and visualizations['3d_volume']:
                story.append(Paragraph("3D Volume Analysis", heading_style))
                try:
                    img_data = base64.b64decode(visualizations['3d_volume'])
                    img_buffer = io.BytesIO(img_data)
                    img = Image(img_buffer, width=7*inch, height=5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    logger.error(f"Error adding 3D visualization: {e}")
            
            story.append(PageBreak())

            story.append(Paragraph("QUANTITATIVE ANALYSIS", title_style))
            story.append(Spacer(1, 0.2*inch))

            charts = self._create_enhanced_visualization_charts(report_data.get('features', {}))
            
            for chart_name, chart_data in charts.items():
                try:
                    img_data = base64.b64decode(chart_data)
                    img_buffer = io.BytesIO(img_data)
                    img = Image(img_buffer, width=7*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3*inch))
                except Exception as e:
                    logger.error(f"Error adding chart {chart_name}: {e}")

            story.append(PageBreak())
            
            disclaimer_style = ParagraphStyle(
                'Disclaimer',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#7f8c8d'),
                alignment=TA_JUSTIFY,
                leftIndent=0.5*inch,
                rightIndent=0.5*inch
            )
            
            story.append(Paragraph("IMPORTANT DISCLAIMERS", heading_style))
            
            disclaimers = [
                "This report is generated using artificial intelligence algorithms for automated brain tumor segmentation and analysis.",
                "The AI model used for report generation is designed to assist healthcare professionals but does not replace clinical judgment.",
                "All quantitative measurements and assessments should be validated by qualified radiologists and medical professionals.",
                "Treatment decisions should not be based solely on this automated analysis.",
                "This system is intended for research and educational purposes and to support clinical decision-making.",
                f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} using {report_data.get('model_used', 'AI system')}."
            ]
            
            for disclaimer in disclaimers:
                story.append(Paragraph(f"• {disclaimer}", disclaimer_style))
                story.append(Spacer(1, 0.1*inch))
            doc.build(story)
            
            logger.info(f"Enhanced PDF report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise