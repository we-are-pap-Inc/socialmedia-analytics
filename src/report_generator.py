"""
PDF Report Generator for Social Media Analytics.

This module generates comprehensive PDF reports for social media analytics data.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    HRFlowable,
    KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker


class SocialMediaReportGenerator:
    """Generate PDF reports for social media analytics."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#4a4a4a'),
            spaceBefore=20,
            spaceAfter=12
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#2a2a2a'),
            spaceBefore=6,
            spaceAfter=6
        ))
        
    def generate_report(
        self,
        instagram_data: Optional[Dict[str, Any]] = None,
        tiktok_data: Optional[Dict[str, Any]] = None,
        comparison_data: Optional[Dict[str, Any]] = None,
        output_path: str = "social_media_report.pdf"
    ):
        """Generate a comprehensive PDF report."""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story
        story = []
        
        # Title page
        story.append(Paragraph(
            "Social Media Analytics Report",
            self.styles['CustomTitle']
        ))
        
        # Get username from data if available
        username = "Unknown"
        if instagram_data and 'profile_info' in instagram_data:
            username = instagram_data['profile_info'].get('username', 'Unknown')
        elif tiktok_data and 'profile_info' in tiktok_data:
            username = tiktok_data['profile_info'].get('username', 'Unknown')
        
        story.append(Paragraph(
            f"@{username}",
            self.styles['Subtitle']
        ))
        
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            self.styles['Normal']
        ))
        
        story.append(Spacer(1, 0.5*inch))
        story.append(HRFlowable(width="80%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['Heading1']))
        
        if instagram_data and tiktok_data:
            summary_data = self._create_summary_table(instagram_data, tiktok_data)
            story.append(summary_data)
        
        story.append(PageBreak())
        
        # Instagram Section
        if instagram_data:
            story.extend(self._create_platform_section("Instagram", instagram_data))
            story.append(PageBreak())
        
        # TikTok Section
        if tiktok_data:
            story.extend(self._create_platform_section("TikTok", tiktok_data))
            story.append(PageBreak())
        
        # Comparison Section
        if comparison_data:
            story.extend(self._create_comparison_section(comparison_data))
        
        # Build PDF
        doc.build(story)
        
    def _create_summary_table(self, instagram_data: Dict, tiktok_data: Dict) -> Table:
        """Create executive summary comparison table."""
        data = [
            ['Metric', 'Instagram', 'TikTok', 'Winner'],
            ['Total Posts', 
             str(instagram_data['total_posts']), 
             str(tiktok_data['total_posts']),
             'TikTok' if tiktok_data['total_posts'] > instagram_data['total_posts'] else 'Instagram'],
            ['Total Engagement', 
             f"{instagram_data['total_engagement']:,}", 
             f"{tiktok_data['total_engagement']:,}",
             'TikTok' if tiktok_data['total_engagement'] > instagram_data['total_engagement'] else 'Instagram'],
            ['Average Views', 
             f"{instagram_data['average_views']:,.0f}", 
             f"{tiktok_data['average_views']:,.0f}",
             'TikTok' if tiktok_data['average_views'] > instagram_data['average_views'] else 'Instagram'],
            ['Engagement Rate', 
             f"{instagram_data['engagement_rate']:.1f}%", 
             f"{tiktok_data['engagement_rate']:.1f}%",
             'Instagram' if instagram_data['engagement_rate'] > tiktok_data['engagement_rate'] else 'TikTok'],
            ['Viral Score', 
             f"{instagram_data['viral_velocity_score']:.1f}", 
             f"{tiktok_data['viral_velocity_score']:.1f}",
             'TikTok' if tiktok_data['viral_velocity_score'] > instagram_data['viral_velocity_score'] else 'Instagram'],
            ['Consistency', 
             f"{instagram_data['content_consistency_index']:.2f}", 
             f"{tiktok_data['content_consistency_index']:.2f}",
             'TikTok' if tiktok_data['content_consistency_index'] > instagram_data['content_consistency_index'] else 'Instagram'],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1*inch])
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            
            # Winner column
            ('TEXTCOLOR', (3, 1), (3, -1), colors.HexColor('#27ae60')),
            ('FONTNAME', (3, 1), (3, -1), 'Helvetica-Bold'),
        ]))
        
        return table
        
    def _create_platform_section(self, platform: str, data: Dict) -> list:
        """Create a section for a specific platform."""
        section = []
        
        # Platform header
        section.append(Paragraph(f"{platform} Analytics", self.styles['Heading1']))
        
        # Add profile information if available
        if 'profile_info' in data and data['profile_info']:
            profile = data['profile_info']
            section.append(Paragraph("Account Information", self.styles['Subtitle']))
            
            profile_data = []
            if profile.get('full_name'):
                profile_data.append(['Full Name', profile['full_name']])
            if profile.get('followers_count') is not None:
                profile_data.append(['Followers', f"{profile['followers_count']:,}"])
            if profile.get('posts_count') is not None:
                profile_data.append(['Total Posts', f"{profile['posts_count']:,}"])
            if profile.get('bio'):
                # Truncate long bios for better display
                bio = profile['bio'][:200] + '...' if len(profile['bio']) > 200 else profile['bio']
                profile_data.append(['Bio', bio])
            
            if profile_data:
                profile_table = Table(profile_data, colWidths=[2*inch, 3.5*inch])
                profile_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
                    ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                ]))
                section.append(profile_table)
                section.append(Spacer(1, 0.3*inch))
        
        # Key metrics
        section.append(Paragraph("Key Performance Metrics", self.styles['Subtitle']))
        
        metrics_data = [
            ['Total Posts', str(data['total_posts'])],
            ['Total Engagement', f"{data['total_engagement']:,}"],
            ['Average Views', f"{data['average_views']:,.0f}"],
            ['Average Likes', f"{data['average_likes']:,.0f}"],
            ['Engagement Rate', f"{data['engagement_rate']:.2f}%"],
            ['Viral Velocity Score', f"{data['viral_velocity_score']:.1f}"],
            ['Content Consistency', f"{data['content_consistency_index']:.2f}"],
            ['Peak Performance Ratio', f"{data['peak_performance_ratio']:.1f}x"],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        section.append(metrics_table)
        section.append(Spacer(1, 0.3*inch))
        
        # Posting frequency
        section.append(Paragraph("Posting Frequency", self.styles['Subtitle']))
        freq = data['posting_frequency']
        section.append(Paragraph(
            f"• {freq['posts_per_day']:.2f} posts per day",
            self.styles['Metric']
        ))
        section.append(Paragraph(
            f"• {freq['avg_hours_between_posts']:.1f} hours between posts",
            self.styles['Metric']
        ))
        section.append(Spacer(1, 0.2*inch))
        
        # Top hashtags
        if 'top_hashtags' in data and data['top_hashtags']:
            section.append(Paragraph("Top Performing Hashtags", self.styles['Subtitle']))
            
            hashtag_data = [['Hashtag', 'Effectiveness Score']]
            for tag, score in list(data['top_hashtags'].items())[:5]:
                if score > 0:
                    hashtag_data.append([f"#{tag}", f"{score:.1f}"])
            
            if len(hashtag_data) > 1:
                hashtag_table = Table(hashtag_data, colWidths=[3*inch, 2*inch])
                hashtag_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
                ]))
                section.append(hashtag_table)
        
        return section
        
    def _create_comparison_section(self, comparison_data: Dict) -> list:
        """Create comparison section between platforms."""
        section = []
        
        section.append(Paragraph("Platform Comparison", self.styles['Heading1']))
        section.append(Paragraph(
            "This section provides a detailed comparison between Instagram and TikTok performance.",
            self.styles['Normal']
        ))
        section.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        section.append(Paragraph("Strategic Recommendations", self.styles['Subtitle']))
        
        recommendations = self._generate_recommendations(comparison_data)
        for rec in recommendations:
            section.append(Paragraph(f"• {rec}", self.styles['Metric']))
        
        return section
        
    def _generate_recommendations(self, data: Dict) -> list:
        """Generate strategic recommendations based on data."""
        recommendations = []
        
        if 'instagram' in data and 'tiktok' in data:
            ig = data['instagram']
            tt = data['tiktok']
            
            # Engagement comparison
            if ig['engagement_rate'] > tt['engagement_rate']:
                recommendations.append(
                    "Instagram shows higher engagement rate - consider increasing content frequency on this platform"
                )
            else:
                recommendations.append(
                    "TikTok demonstrates superior engagement - prioritize TikTok content creation"
                )
            
            # Viral potential
            if tt['viral_velocity_score'] > 8:
                recommendations.append(
                    "High viral velocity on TikTok indicates strong potential for viral content"
                )
            
            # Consistency
            if ig['content_consistency_index'] < 0.7 or tt['content_consistency_index'] < 0.7:
                recommendations.append(
                    "Improve posting consistency to maintain audience engagement"
                )
            
            # Hashtag usage
            if 'top_hashtags' in ig:
                active_hashtags = sum(1 for score in ig['top_hashtags'].values() if score > 0)
                if active_hashtags < 10:
                    recommendations.append(
                        "Increase hashtag diversity on Instagram (currently using limited hashtags)"
                    )
        
        return recommendations


def generate_comprehensive_report(
    instagram_metrics_path: Optional[str] = None,
    tiktok_metrics_path: Optional[str] = None,
    comparison_path: Optional[str] = None,
    output_path: str = "wifeygpt_analytics_report.pdf"
):
    """Generate a comprehensive PDF report from existing data files."""
    
    # Load data
    instagram_data = None
    tiktok_data = None
    comparison_data = None
    
    if instagram_metrics_path and Path(instagram_metrics_path).exists():
        if instagram_metrics_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(instagram_metrics_path)
            instagram_data = df.to_dict('records')[0] if len(df) > 0 else None
            # Reconstruct nested dictionaries from flattened CSV
            if instagram_data:
                instagram_data['posting_frequency'] = {
                    'posts_per_day': instagram_data.get('posting_frequency.posts_per_day', 0),
                    'posts_per_week': instagram_data.get('posting_frequency.posts_per_week', 0),
                    'posts_per_month': instagram_data.get('posting_frequency.posts_per_month', 0),
                    'avg_hours_between_posts': instagram_data.get('posting_frequency.avg_hours_between_posts', 0),
                }
                # Extract top hashtags
                instagram_data['top_hashtags'] = {}
                for key in list(instagram_data.keys()):
                    if key.startswith('top_hashtags.'):
                        tag_name = key.replace('top_hashtags.', '')
                        instagram_data['top_hashtags'][tag_name] = instagram_data[key]
        elif instagram_metrics_path.endswith('.json'):
            with open(instagram_metrics_path, 'r') as f:
                instagram_data = json.load(f)
    
    if tiktok_metrics_path and Path(tiktok_metrics_path).exists():
        if tiktok_metrics_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(tiktok_metrics_path)
            tiktok_data = df.to_dict('records')[0] if len(df) > 0 else None
            # Reconstruct nested dictionaries from flattened CSV
            if tiktok_data:
                tiktok_data['posting_frequency'] = {
                    'posts_per_day': tiktok_data.get('posting_frequency.posts_per_day', 0),
                    'posts_per_week': tiktok_data.get('posting_frequency.posts_per_week', 0),
                    'posts_per_month': tiktok_data.get('posting_frequency.posts_per_month', 0),
                    'avg_hours_between_posts': tiktok_data.get('posting_frequency.avg_hours_between_posts', 0),
                }
                # Extract top hashtags
                tiktok_data['top_hashtags'] = {}
                for key in list(tiktok_data.keys()):
                    if key.startswith('top_hashtags.'):
                        tag_name = key.replace('top_hashtags.', '')
                        tiktok_data['top_hashtags'][tag_name] = tiktok_data[key]
        elif tiktok_metrics_path.endswith('.json'):
            with open(tiktok_metrics_path, 'r') as f:
                tiktok_data = json.load(f)
    
    if comparison_path and Path(comparison_path).exists():
        with open(comparison_path, 'r') as f:
            comparison_data = json.load(f)
    
    # Generate report
    generator = SocialMediaReportGenerator()
    generator.generate_report(
        instagram_data=instagram_data,
        tiktok_data=tiktok_data,
        comparison_data=comparison_data,
        output_path=output_path
    )
    
    return output_path


def generate_multi_account_report(
    all_results: list,
    output_path: str = "multi_account_report.pdf"
):
    """Generate a comprehensive report for multiple accounts."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    generator = SocialMediaReportGenerator()
    story = []
    
    # Title page
    story.append(Paragraph(
        "Multi-Account Social Media Analytics",
        generator.styles['CustomTitle']
    ))
    
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        generator.styles['Normal']
    ))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Summary statistics
    total_posts = sum(r['metrics'].get('total_posts', 0) for r in all_results if r)
    total_engagement = sum(r['metrics'].get('total_engagement', 0) for r in all_results if r)
    total_views = sum(r['metrics'].get('average_views', 0) * r['metrics'].get('total_posts', 0) 
                     for r in all_results if r)
    
    summary_data = [
        ['Total Accounts', str(len(all_results))],
        ['Total Posts', f"{total_posts:,}"],
        ['Total Engagement', f"{total_engagement:,}"],
        ['Total Views', f"{int(total_views):,}"],
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    
    story.append(Paragraph("Overall Statistics", generator.styles['Heading1']))
    story.append(summary_table)
    story.append(PageBreak())
    
    # Individual account summaries
    story.append(Paragraph("Account Performance Summary", generator.styles['Heading1']))
    
    # Create comparison table
    comparison_data = [['Platform', 'Account', 'Posts', 'Engagement', 'Avg Views', 'Eng Rate']]
    
    for result in all_results:
        if result and 'metrics' in result:
            metrics = result['metrics']
            comparison_data.append([
                result['platform'].title(),
                f"@{result['username']}",
                str(metrics.get('total_posts', 0)),
                f"{metrics.get('total_engagement', 0):,}",
                f"{metrics.get('average_views', 0):,.0f}",
                f"{metrics.get('engagement_rate', 0):.1f}%"
            ])
    
    comp_table = Table(comparison_data, colWidths=[1.2*inch, 1.5*inch, 0.8*inch, 1.2*inch, 1.2*inch, 0.8*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    
    story.append(comp_table)
    
    # Build PDF
    doc.build(story)
    return output_path